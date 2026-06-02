"""
act_inference.py
----------------
Runs the trained ACT policy in CoppeliaSim to evaluate pick-and-place success.

KEY DIFFERENCE FROM BC INFERENCE:
  ACT predicts a CHUNK of actions at once (chunk_size=10).
  The chunk is executed step-by-step without re-querying the policy.
  The policy is only called again after the full chunk is executed.
  This breaks the BC feedback loop that caused freezing.

HOW GRASPING WORKS:
  Same as BC — position-based triggers mirror the Lua script logic:
    - attach cube when EE is close enough to cube (setObjectParent)
    - detach cube when EE is close enough to place pose

BEFORE RUNNING:
  1. Disable the UR5 thread script in CoppeliaSim.
  2. Keep the BarrettHand script enabled.
  3. Put act_best.pth and the four .npy stats files in --stats_dir.

Usage:
    python act_inference.py --n_episodes 20
    python act_inference.py --checkpoint ../checkpoints/act_checkpoints/act_best.pth
                            --stats_dir  ../checkpoints/act_checkpoints/
                            --n_episodes 20
"""

import os
import sys
import argparse
import time
import random
import json
import math

import numpy as np
import torch
import torch.nn as nn

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(PROJECT_ROOT)

from scripts.utils.zmq_remoteapi_connection_utils import connect_to_coppeliasim
from scripts.utils.randomize_cube_position_utils import randomize_cube_position


# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════
OBS_DIM    = 17
ACTION_DIM = 6
HIDDEN_DIM = 256
CHUNK_SIZE = 10    # must match training

MAX_STEPS  = 600   # max total steps per episode
N_EPISODES = 20

# cube randomisation — same range as training
CUBE_X_RANGE = (0.6,   1.1)
CUBE_Y_RANGE = (-0.78, -1.0)
CUBE_Z_FIXED = 0.38

# grasp trigger
GRASP_XY_THRESHOLD = 0.06
GRASP_Z_THRESHOLD  = 0.08

# place trigger
PLACE_XY_THRESHOLD = 0.06
PLACE_Z_THRESHOLD  = 0.05

# success check
PLACE_TABLE_Z        = 0.30
PLACE_Z_TOL          = 0.05
SUCCESS_XY_THRESHOLD = 0.12

# stuck detection
STUCK_WINDOW    = 50
STUCK_THRESHOLD = 0.005


# ══════════════════════════════════════════════════════════════════════════════
# MODEL — must exactly match act_training.py
# ══════════════════════════════════════════════════════════════════════════════
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe  = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float()
                        * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return self.dropout(x + self.pe[:, :x.size(1)])


class ACTPolicy(nn.Module):
    def __init__(self, obs_dim=OBS_DIM, action_dim=ACTION_DIM,
                 chunk_size=CHUNK_SIZE, d_model=256, nhead=8,
                 num_encoder_layers=4, num_decoder_layers=4,
                 dim_feedforward=1024, dropout=0.1):
        super().__init__()
        self.chunk_size = chunk_size
        self.d_model    = d_model

        self.obs_proj = nn.Sequential(
            nn.Linear(obs_dim, d_model),
            nn.LayerNorm(d_model),
        )
        self.action_queries = nn.Embedding(chunk_size, d_model)
        self.pos_enc        = PositionalEncoding(d_model, dropout=dropout)

        enc_layer    = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_encoder_layers)

        dec_layer    = nn.TransformerDecoderLayer(
            d_model, nhead, dim_feedforward, dropout, batch_first=True)
        self.decoder = nn.TransformerDecoder(dec_layer, num_decoder_layers)

        self.action_head = nn.Linear(d_model, action_dim)

    def forward(self, obs):
        B       = obs.size(0)
        obs_emb = self.obs_proj(obs).unsqueeze(1)
        memory  = self.encoder(obs_emb)
        idx     = torch.arange(self.chunk_size, device=obs.device)
        queries = self.action_queries(idx).unsqueeze(0).expand(B, -1, -1)
        queries = self.pos_enc(queries)
        decoded = self.decoder(queries, memory)
        return self.action_head(decoded)   # (B, chunk_size, action_dim)


# ══════════════════════════════════════════════════════════════════════════════
# SETUP
# ══════════════════════════════════════════════════════════════════════════════
def load_policy(checkpoint_path, stats_dir, device):
    model = ACTPolicy().to(device)
    model.eval()

    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    print(f"Loaded checkpoint — epoch {ckpt['epoch']}  "
          f"val_loss={ckpt['val_loss']:.6f}  "
          f"chunk_size={ckpt['chunk_size']}")

    obs_mean = torch.tensor(
        np.load(os.path.join(stats_dir, 'act_obs_mean.npy')),
        dtype=torch.float32).to(device)
    obs_std  = torch.tensor(
        np.load(os.path.join(stats_dir, 'act_obs_std.npy')),
        dtype=torch.float32).to(device)
    act_mean = torch.tensor(
        np.load(os.path.join(stats_dir, 'act_act_mean.npy')),
        dtype=torch.float32).to(device)
    act_std  = torch.tensor(
        np.load(os.path.join(stats_dir, 'act_act_std.npy')),
        dtype=torch.float32).to(device)

    return model, obs_mean, obs_std, act_mean, act_std


def get_handles(sim):
    cube_handle   = sim.getObject('/Cube')
    tip_handle    = sim.getObject('/UR5/BarrettHand/tip')
    attach_handle = sim.getObject('/UR5/BarrettHand/attachPoint')
    place_handle  = sim.getObject('/placePose')
    joint_handles = [sim.getObject('/UR5/joint', {'index': i}) for i in range(6)]
    finger_joint  = sim.getObject('/UR5/BarrettHand/jointC_2')
    hand_script   = sim.getObject('/UR5/BarrettHand/Script')
    return (cube_handle, tip_handle, attach_handle,
            place_handle, joint_handles, finger_joint, hand_script)


# ══════════════════════════════════════════════════════════════════════════════
# OBSERVATION
# ══════════════════════════════════════════════════════════════════════════════
def get_observation(sim, cube_handle, tip_handle, joint_handles, finger_joint):
    jp   = np.array([sim.getJointPosition(j) for j in joint_handles])
    ee   = np.array(sim.getObjectPose(tip_handle, -1))
    cube = np.array(sim.getObjectPosition(cube_handle, -1))
    grip = np.array([sim.getJointPosition(finger_joint)])
    return np.concatenate([jp, ee, cube, grip])   # (17,)


# ══════════════════════════════════════════════════════════════════════════════
# ACTION CHUNK PREDICTION
# ══════════════════════════════════════════════════════════════════════════════
def predict_chunk(obs_np, model, obs_mean, obs_std, act_mean, act_std, device):
    """
    Normalise obs → run ACT → denormalise → return full action chunk.
    Returns: np.array of shape (chunk_size, action_dim)
    """
    obs_t  = torch.tensor(obs_np, dtype=torch.float32).unsqueeze(0).to(device)
    obs_t  = (obs_t - obs_mean) / obs_std

    with torch.no_grad():
        chunk_norm = model(obs_t)   # (1, chunk_size, action_dim)

    chunk = (chunk_norm.squeeze(0) * act_std + act_mean).cpu().numpy()
    return chunk   # (chunk_size, action_dim)


# ══════════════════════════════════════════════════════════════════════════════
# GRIPPER HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def attach_cube(sim, cube_handle, attach_handle, hand_script):
    sim.setObjectParent(cube_handle, attach_handle, True)
    sim.callScriptFunction('closeHand', hand_script)
    print("    [grasp] cube attached")


def detach_cube(sim, cube_handle, hand_script):
    sim.setObjectParent(cube_handle, -1, True)
    sim.callScriptFunction('openHand', hand_script)
    print("    [release] cube detached")


# ══════════════════════════════════════════════════════════════════════════════
# EPISODE
# ══════════════════════════════════════════════════════════════════════════════
def run_episode(sim, model, obs_mean, obs_std, act_mean, act_std, device,
                cube_handle, tip_handle, attach_handle, place_handle,
                joint_handles, finger_joint, hand_script, ep_idx):

    cube_start = randomize_cube_position(sim, cube_handle)
    place_pos  = np.array(sim.getObjectPosition(place_handle, -1))

    sim.clearInt32Signal('pick_and_place_done')
    sim.startSimulation()
    time.sleep(0.3)

    grasped        = False
    cube_lifted    = False
    released       = False
    success        = False
    failure_reason = None
    final_cube     = np.array(cube_start, dtype=float)

    # stuck detection
    ee_history  = []

    # ACT chunk buffer — actions queued to execute
    chunk_buffer = []   # list of (action_dim,) arrays
    step         = 0

    steps_to_grasp   = None
    steps_to_release = None
    xy_dist_final    = None

    while step < MAX_STEPS:

        # ── sim crash check ───────────────────────────────────────────────
        if sim.getSimulationState() == sim.simulation_stopped:
            failure_reason = "sim_stopped"
            print(f"    [warn] sim stopped at step {step}")
            break

        # ── refill chunk buffer when empty ───────────────────────────────
        # This is the core ACT inference loop:
        #   query policy → get 10 actions → execute them one by one
        #   only re-query when the buffer is empty
        if len(chunk_buffer) == 0:
            obs         = get_observation(sim, cube_handle, tip_handle,
                                          joint_handles, finger_joint)
            chunk       = predict_chunk(obs, model, obs_mean, obs_std,
                                        act_mean, act_std, device)
            chunk_buffer = list(chunk)   # 10 actions queued

        # ── execute next action from buffer ───────────────────────────────
        action = chunk_buffer.pop(0)   # take first, rest stay queued
        action = np.clip(action, -np.pi, np.pi)
        for j, jh in enumerate(joint_handles):
            sim.setJointTargetPosition(jh, float(action[j]))
        time.sleep(0.05)

        step += 1

        # ── read current state for trigger checks ─────────────────────────
        obs      = get_observation(sim, cube_handle, tip_handle,
                                   joint_handles, finger_joint)
        ee_pos   = obs[6:9]
        cube_pos = obs[13:16]

        # ── stuck detection ───────────────────────────────────────────────
        ee_history.append(ee_pos.copy())
        if len(ee_history) > STUCK_WINDOW:
            movement = np.linalg.norm(
                ee_history[-1] - ee_history[-STUCK_WINDOW]
            )
            if movement < STUCK_THRESHOLD:
                failure_reason = "stuck"
                print(f"    [stuck] step={step}  movement={movement:.4f}m")
                break

        # ── grasp trigger ─────────────────────────────────────────────────
        if not grasped:
            xy_dist = np.linalg.norm(ee_pos[:2] - cube_pos[:2])
            z_dist  = abs(ee_pos[2] - cube_pos[2])
            if xy_dist < GRASP_XY_THRESHOLD and z_dist < GRASP_Z_THRESHOLD:
                attach_cube(sim, cube_handle, attach_handle, hand_script)
                if sim.getObjectParent(cube_handle) == attach_handle:
                    grasped     = True
                    cube_lifted = True
                    steps_to_grasp   = step
                    chunk_buffer.clear()   # discard remaining chunk
                    time.sleep(0.5)        # let grasp settle

        # ── place trigger ──────────────────────────────────────────────────
        elif grasped and not released:
            xy_dist = np.linalg.norm(ee_pos[:2] - place_pos[:2])
            z_dist  = abs(ee_pos[2] - place_pos[2])
            if xy_dist < PLACE_XY_THRESHOLD and z_dist < PLACE_Z_THRESHOLD:
                detach_cube(sim, cube_handle, hand_script)
                released = True
                steps_to_release = step
                chunk_buffer.clear()   # discard remaining chunk
                time.sleep(0.5)

        # ── success check ──────────────────────────────────────────────────
        elif released:
            final_cube    = np.array(sim.getObjectPosition(cube_handle, -1))
            xy_dist_final = np.linalg.norm(final_cube[:2] - place_pos[:2])
            height_ok     = abs(final_cube[2] - PLACE_TABLE_Z) < PLACE_Z_TOL
            if xy_dist_final < SUCCESS_XY_THRESHOLD and height_ok:
                success = True
                break

    # always read final cube position
    try:
        final_cube = np.array(sim.getObjectPosition(cube_handle, -1))
    except Exception:
        pass

    # failure classification
    if not success and failure_reason is None:
        if not grasped:
            failure_reason = "never_grasped"
        elif not released:
            failure_reason = "grasped_but_never_released"
        else:
            failure_reason = "released_but_misplaced"

    # cleanup — detach cube if still attached
    if grasped and not released:
        try:
            sim.setObjectParent(cube_handle, -1, True)
        except Exception:
            pass

    sim.stopSimulation()
    while sim.getSimulationState() != sim.simulation_stopped:
        time.sleep(0.1)

    return {
        'episode':          ep_idx,
        'success':          success,
        'failure_reason':   failure_reason,
        'grasped':          grasped,
        'released':         released,
        'steps':            step,
        'steps_to_grasp':   steps_to_grasp,    
        'steps_to_release': steps_to_release,   
        'placement_error_xy': xy_dist_final,     
        'cube_start':       list(cube_start),
        'cube_final':       final_cube.tolist(),
        'place_pos':        place_pos.tolist(),
    }


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
def run_evaluation(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device  : {device}")

    model, obs_mean, obs_std, act_mean, act_std = load_policy(
        args.checkpoint, args.stats_dir, device)

    sim = connect_to_coppeliasim()
    (cube_handle, tip_handle, attach_handle, place_handle,
     joint_handles, finger_joint, hand_script) = get_handles(sim)

    n       = args.n_episodes
    results = []

    print(f"\nRunning {n} evaluation episodes\n")
    print(f"{'Ep':>4} | {'Steps':>6} | {'Grasped':>8} | "
          f"{'Released':>9} | {'Result':>8} | Failure reason")
    print("-" * 70)

    for ep in range(1, n + 1):
        r = run_episode(
            sim, model, obs_mean, obs_std, act_mean, act_std, device,
            cube_handle, tip_handle, attach_handle, place_handle,
            joint_handles, finger_joint, hand_script, ep
        )
        results.append(r)

        result_str = 'SUCCESS' if r['success'] else 'FAILURE'
        reason     = r['failure_reason'] or '-'
        print(f"{ep:>4} | {r['steps']:>6} | {str(r['grasped']):>8} | "
              f"{str(r['released']):>9} | {result_str:>8} | {reason}")

    # summary
    n_success  = sum(r['success']  for r in results)
    n_grasped  = sum(r['grasped']  for r in results)
    n_released = sum(r['released'] for r in results)
    avg_steps  = np.mean([r['steps'] for r in results])

    # averages only over episodes where the event occurred
    grasp_steps   = [r['steps_to_grasp']     for r in results if r['steps_to_grasp']    is not None]
    release_steps = [r['steps_to_release']   for r in results if r['steps_to_release']  is not None]
    place_errors  = [r['placement_error_xy'] for r in results if r['placement_error_xy'] is not None]

    from collections import Counter
    failure_counts = Counter(
        r['failure_reason'] for r in results if r['failure_reason']
    )

    print("\n" + "=" * 70)
    print("EVALUATION RESULTS — ACT Policy")
    print(f"  Episodes       : {n}")
    print(f"  Grasped        : {n_grasped} / {n}  ({n_grasped/n*100:.1f}%)")
    print(f"  Released       : {n_released} / {n}  ({n_released/n*100:.1f}%)")
    print(f"  Full success   : {n_success} / {n}  ({n_success/n*100:.1f}%)")
    print(f"  Avg steps      : {avg_steps:.1f}")
    print(f"  Avg grasp steps: {np.mean(grasp_steps):.1f} ± {np.std(grasp_steps):.1f}")
    print(f"  Avg release steps: {np.mean(release_steps):.1f} ± {np.std(release_steps):.1f}")
    print(f"  Avg place errors: {np.mean(place_errors):.1f} ± {np.std(place_errors):.1f}")
    print(f"\n  Failure breakdown:")
    for k, v in failure_counts.items():
        print(f"    {k}: {v}")
    print("=" * 70)

    out = {
        'policy':                 'ACT',
        'n_episodes':             n,
        'n_grasped':              n_grasped,
        'n_released':             n_released,
        'n_success':              n_success,
        'success_rate':           n_success / n * 100,
        'grasp_rate':             n_grasped / n * 100,
        'release_rate':           n_released / n * 100,
        'completion_rate':        n_grasped  / n * 100,
        'avg_steps':              avg_steps,
        'avg_steps_to_grasp':     float(np.mean(grasp_steps))   if grasp_steps   else None, 
        'avg_steps_to_release':   float(np.mean(release_steps)) if release_steps else None,  
        'avg_placement_error_xy': float(np.mean(place_errors))  if place_errors  else None, 
        'failure_breakdown':      dict(failure_counts),
        'episodes':               results,
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.save_results)),
                exist_ok=True)
    with open(args.save_results, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\nResults saved to: {args.save_results}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint',   type=str,
                        default='./checkpoints/act/act_best.pth')
    parser.add_argument('--stats_dir',    type=str,
                        default='./checkpoints/act/')
    parser.add_argument('--n_episodes',   type=int, default=N_EPISODES)
    parser.add_argument('--save_results', type=str,
                        default='../results/act_eval_results.json')
    args = parser.parse_args()
    run_evaluation(args)