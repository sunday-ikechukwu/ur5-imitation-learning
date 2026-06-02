""" bc_inference.py 
--------------- 
Runs the trained BC policy in CoppeliaSim to evaluate pick-and-place success. 

HOW GRASPING WORKS IN THIS SCENE:
  The BarrettHand does not use physics-based grasping. The Lua expert script
  attaches the cube to the gripper via sim.setObjectParent() when the arm
  reaches the pick pose, and detaches it at the place pose. This script
  replicates that same logic — the BC policy controls the 6 arm joints,
  and position-based triggers handle attach/detach.

The BC policy: 
    1. Reads the current observation from CoppeliaSim 
    2. Normalises it using the saved training stats 
    3. Passes it through the MLP to get target joint angles 
    4. Sends those target angles to the robot via setJointTargetPosition

BEFORE RUNNING:
  1. Disable the UR5 thread script in CoppeliaSim (the arm controller Lua).
     The BarrettHand script can stay — it handles finger animation.
  2. Make sure bc_best.pth and the four .npy stats files are in --stats_dir.
       
Success criteria: 
    - Cube z position drops below DROP_Z_THRESHOLD (cube was lifted and placed) - AND cube has moved from its start position by at least MIN_XY_DISPLACEMENT 

Failure criteria: 
    - Episode exceeds MAX_STEPS without success 

Usage: 
    python bc_inference.py --checkpoint ./checkpoints/bc/bc_best.pth --stats_dir ./checkpoints/bc/ --n_episodes 20 """ 

import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(PROJECT_ROOT)

import argparse
import random
import time 
import json

import numpy as np 
import torch 
import torch.nn as nn 

from scripts.utils.zmq_remoteapi_connection_utils import connect_to_coppeliasim 
from scripts.utils.pick_and_place_imitation_data_recorder_utils import get_handles
from scripts.utils.randomize_cube_position_utils import randomize_cube_position


#══════════════════════════════════════════════════════════════════════════════ # CONFIG — tweak these to match your scene #══════════════════════════════════════════════════════════════════════════════
OBS_DIM = 17 
ACTION_DIM = 6 
HIDDEN_DIM = 256 

STEP_SLEEP = 0.05 # seconds between policy steps (matches 20 Hz training
MAX_STEPS = 600 # max timesteps per episode before declaring failure 
N_EPISODES = 20 # default — overridden by --n_episodes arg 


# grasp trigger — attach cube when EE is within this XY distance of the cube
# and below this Z height (arm has descended to pick level)
GRASP_XY_THRESHOLD = 0.06   # metres
GRASP_Z_THRESHOLD  = 0.08   # EE z must be below this to trigger grasp

# place trigger — detach cube when EE is within this XY distance of place pose
PLACE_XY_THRESHOLD =  0.06  # metres 
PLACE_Z_THRESHOLD  =  0.05   # EE z must be below this to trigger place  

PLACE_TABLE_Z = 0.30
PLACE_Z_TOL   = 0.02

# success = cube ended up near the place pose after detach
SUCCESS_XY_THRESHOLD = 0.12  # metres — cube final pos vs place pos

# stuck detection
STUCK_WINDOW    = 50 # number of steps to look back when checking if stuck
STUCK_THRESHOLD = 0.005

FIXED_CUBE_POSITIONS = [
    [0.6498418921658287,-0.7675464577815916,0.38],
    [0.8923649103848641,-0.7611502363630376,0.38],
    [0.6002334666697562,-0.8637614458475129,0.38],
    [0.4788798132287051,-1.0857315762586333,0.38],
    [1.016830102375254,-0.8461788481528731,0.38],
    [0.7443516268989547,-0.8630068775851197,0.38],
    [0.5040945881560177,-1.0115742054803956,0.38],
    [0.7523312456148699,-0.8153439980898589,0.38],
    [1.0019560232385152,-0.8976649760568642,0.38],
    [0.5236647715350743,-0.8249247936951015,0.38],
    [1.0154704469481348,-1.0702890744238966,0.38],
    [1.0436107123114742,-0.8333175656841623,0.38],
    [0.4640781827827326,-1.0089309478812134,0.38],
    [0.9945538311277629,-1.0667613057272338,0.38],
    [0.5178485953663181,-0.9264881535865996,0.38],
    [1.051032693267633,-0.9891749928655511,0.38],
    [0.88351111057389,-0.795815524256971,0.38],
    [0.8380967575014645,-0.9082975059838421,0.38],
    [0.6961241471424688,-0.7995853187581607,0.38],
    [0.9645542589671192,-0.9415727472628591,0.38]
]

#══════════════════════════════════════════════════════════════════════════════ # MODEL (must match bc_training.py exactly) #══════════════════════════════════════════════════════════════════════════════ 
class BCPolicy(nn.Module): 
    def __init__(self, obs_dim=OBS_DIM, action_dim=ACTION_DIM, hidden_dim=HIDDEN_DIM): 
        super().__init__() 
        self.net = nn.Sequential( 
            nn.Linear(obs_dim, hidden_dim), 
            nn.LayerNorm(hidden_dim), 
            nn.ReLU(), 
            nn.Dropout(0.1), 

            nn.Linear(hidden_dim, hidden_dim), 
            nn.LayerNorm(hidden_dim), 
            nn.ReLU(), 
            nn.Dropout(0.1), 

            nn.Linear(hidden_dim, hidden_dim // 2), 
            nn.LayerNorm(hidden_dim // 2), 
            nn.ReLU(), 

            nn.Linear(hidden_dim // 2, action_dim), 
        ) 

    def forward(self, obs): 
        return self.net(obs) # 

#══════════════════════════════════════════════════════════════════════════════ # SETUP #══════════════════════════════════════════════════════════════════════════════
def load_policy(checkpoint_path, stats_dir, device): 
    """Load BC model and normalisation stats.""" 
    model = BCPolicy().to(device) 
     
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False) 
    model.load_state_dict(ckpt['model_state_dict']) 

    model.eval()
    print(f"Loaded checkpoint — epoch {ckpt['epoch']} val_loss={ckpt['val_loss']:.6f}") 

    obs_mean = torch.tensor(np.load(os.path.join(stats_dir, 'obs_mean.npy')), dtype=torch.float32).to(device) 
    obs_std = torch.tensor(np.load(os.path.join(stats_dir, 'obs_std.npy')), dtype=torch.float32).to(device) 
    act_mean = torch.tensor(np.load(os.path.join(stats_dir, 'act_mean.npy')), dtype=torch.float32).to(device) 
    act_std = torch.tensor(np.load(os.path.join(stats_dir, 'act_std.npy')), dtype=torch.float32).to(device) 
    return model, obs_mean, obs_std, act_mean, act_std 

 
#══════════════════════════════════════════════════════════════════════════════ # OBSERVATION / ACTION #══════════════════════════════════════════════════════════════════════════════
def get_observation(sim, cube_handle, tip_handle, joint_handles, finger_joint):
    """Read current state from CoppeliaSim and return as numpy array (17,)."""
    jp = np.array([sim.getJointPosition(j) for j in joint_handles]) # (6,) 
    ee = np.array(sim.getObjectPose(tip_handle, -1)) # (7,) 
    cube = np.array(sim.getObjectPosition(cube_handle, -1)) # (3,) 
    grip = np.array([sim.getJointPosition(finger_joint)]) # (1,) 
    return np.concatenate([jp, ee, cube, grip]) # (17,) 

def predict_action(obs_np, model, obs_mean, obs_std, act_mean, act_std, device):
    """Normalise obs → run policy → denormalise → return joint targets (6,)."""
    obs_t = torch.tensor(obs_np, dtype=torch.float32).unsqueeze(0).to(device)
    obs_t = (obs_t - obs_mean) / (obs_std + 1e-8) 
    
    with torch.no_grad(): 
        action_norm = model(obs_t) # (1, 6) normalised 
    action = (action_norm * act_std + act_mean).squeeze(0).cpu().numpy() # (6,) 
    return action 
    
#══════════════════════════════════════════════════════════════════════════════ # GRIPPER STATE HELPERS #══════════════════════════════════════════════════════════════════════════════
def attach_cube(sim, cube_handle, attach_handle, hand_script):
    """Mirror what the Lua script does: parent cube to gripper attach point."""
    sim.setObjectParent(cube_handle, attach_handle, True)
    sim.callScriptFunction('closeHand', hand_script)
    print("    [grasp] cube attached to gripper")

def detach_cube(sim, cube_handle, hand_script):
    """Mirror what the Lua script does: unparent cube and open hand."""
    sim.setObjectParent(cube_handle, -1, True)
    sim.callScriptFunction('openHand', hand_script)
    print("    [release] cube detached from gripper")

#══════════════════════════════════════════════════════════════════════════════ #EPISODE
#══════════════════════════════════════════════════════════════════════════════
def run_episode(sim, model, obs_mean, obs_std, act_mean, act_std, device,
                cube_handle, tip_handle, joint_handles, finger_joint, attach_handle, place_handle, hand_script, ep_idx):
 
    # cube_start = randomize_cube_position(sim, cube_handle)
    pos = FIXED_CUBE_POSITIONS[ep_idx - 1]
    sim.setObjectPosition(cube_handle, -1, pos)
    cube_start = np.array(pos)
    place_pos  = np.array(sim.getObjectPosition(place_handle, -1))
 
    sim.clearInt32Signal('pick_and_place_done')
    sim.startSimulation()
    time.sleep(0.3)   # let physics settle
 
    grasped    = False
    cube_lifted = False
    released   = False
    success    = False
    failure_reason = None
    final_cube = np.array(cube_start, dtype=float)

    # stuck detection
    ee_history  = []

    steps_to_grasp   = None
    steps_to_release = None
    xy_dist_final    = None

 
    for step in range(MAX_STEPS):
 
        # ── sim crash check ───────────────────────────────────────────────
        if sim.getSimulationState() == sim.simulation_stopped:
            failure_reason = "sim_stopped"
            print(f"    [warn] sim stopped at step {step}")
            break

        # ── observe ──────────────────────────────────────────────────────────
        obs = get_observation(sim, cube_handle, tip_handle,
                              joint_handles, finger_joint)
        ee_pos   = obs[6:9]    # EE position from the pose slice
        cube_pos = obs[13:16]  # cube position slice
 
        # ── predict & act ─────────────────────────────────────────────────
        action = predict_action(obs, model, obs_mean, obs_std,
                                act_mean, act_std, device)
        for j, jh in enumerate(joint_handles):
            sim.setJointTargetPosition(jh, float(action[j])) #sim.setJointTargetPosition
 
        time.sleep(STEP_SLEEP)

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
        # attach cube when EE is close to cube in XY and low enough in Z
        if not grasped:
            xy_dist_to_cube = np.linalg.norm(ee_pos[:2] - cube_pos[:2])
            z_dist_to_cube = abs(ee_pos[2] - cube_pos[2])

            if (
                xy_dist_to_cube < GRASP_XY_THRESHOLD
                and z_dist_to_cube < GRASP_Z_THRESHOLD
            ):
                attach_cube(sim, cube_handle, attach_handle, hand_script)

                parent = sim.getObjectParent(cube_handle)
                if parent == attach_handle:
                    grasped = True
                    cube_lifted = True
                    steps_to_grasp   = step + 1
                    time.sleep(0.5)   # brief pause — mirrors Lua's sim.wait(0.5)
 
        # ── place trigger ──────────────────────────────────────────────────
        # detach cube when EE is close to place pose (and we're holding the cube)
        elif grasped and not released:
            xy_dist_to_place = np.linalg.norm(ee_pos[:2] - place_pos[:2])
            z_dist_to_place = abs(ee_pos[2] - place_pos[2])
            if (
                xy_dist_to_place < PLACE_XY_THRESHOLD
                and z_dist_to_place < PLACE_Z_THRESHOLD
            ):
                detach_cube(sim, cube_handle, hand_script)
                released = True
                steps_to_release = step + 1
                time.sleep(0.5)   # brief pause — mirrors Lua's sim.wait(0.5)
 
        # ── success check ──────────────────────────────────────────────────
        # after releasing, check cube landed near the place pose
        elif released:
            final_cube = np.array(sim.getObjectPosition(cube_handle, -1)) 
            xy_dist_final = np.linalg.norm(final_cube[:2] - place_pos[:2])
            cube_height_ok = abs(final_cube[2] - PLACE_TABLE_Z) < PLACE_Z_TOL
            if (
                released
                and cube_lifted
                and xy_dist_final < SUCCESS_XY_THRESHOLD
                and cube_height_ok
            ):
                success = True
                break

    # always read final cube position
    try:
        final_cube = np.array(sim.getObjectPosition(cube_handle, -1))
    except Exception:
        pass

    # ─────────────────────────────────────────────────────────────────────
    # FINAL FAILURE REASON CLASSIFICATION
    # ─────────────────────────────────────────────────────────────────────

    if success:
        failure_reason = None

    else:
        if failure_reason is None:  # not sim crash

            if not grasped:
                failure_reason = "never_grasped"

            elif grasped and not released:
                failure_reason = "grasped_but_never_released"

            elif released and not success:
                failure_reason = "released_but_misplaced"

            else:
                failure_reason = "timeout"

    # cleanup
    # if cube still attached (failed mid-grasp), detach before stopping
    if grasped and not released:
        try:
            sim.setObjectParent(cube_handle, -1, True)
        except Exception:
            pass
 
    sim.stopSimulation()
    while sim.getSimulationState() != sim.simulation_stopped:
        time.sleep(0.1)
 
    return {
        'episode':        ep_idx,
        'success':        success,
        'failure_reason': failure_reason,
        'grasped':        grasped,
        'released':       released,
        'steps':          step + 1,
        'steps_to_grasp':   steps_to_grasp,    
        'steps_to_release': steps_to_release,   
        'placement_error_xy': xy_dist_final,
        'cube_start':     cube_start.tolist(),
        'cube_final':     final_cube.tolist(),
        'place_pos':      place_pos.tolist()
    }

#══════════════════════════════════════════════════════════════════════════════ # MAIN EVALUATION LOOP #══════════════════════════════════════════════════════════════════════════════
def run_evaluation(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device  : {device}")
 
    model, obs_mean, obs_std, act_mean, act_std = load_policy(
        args.checkpoint, args.stats_dir, device)
 
    sim = connect_to_coppeliasim()
    (
        cube_handle,
        tip_handle,
        joint_handles,
        finger_joint,
        attach_handle,
        place_handle,
        hand_script
    ) = get_handles(sim)
 
    n  = args.n_episodes
    results = []
 
    print(f"\nRunning {n} evaluation episodes\n")
    print(f"{'Ep':>4} | {'Steps':>6} | {'Grasped':>8} | "
          f"{'Released':>9} | {'Result':>8}")
    print("-" * 70)
 
    for ep in range(1, n + 1):
        r = run_episode(
            sim,
            model,
            obs_mean,
            obs_std,
            act_mean,
            act_std,
            device,
            cube_handle,
            tip_handle,
            joint_handles,
            finger_joint,
            attach_handle,
            place_handle,
            hand_script,
            ep
        )
        results.append(r)
 
        result_str = 'SUCCESS' if r['success'] else 'FAILURE'
        print(f"{ep:>4} | {r['steps']:>6} | {str(r['grasped']):>8} | "
              f"{str(r['released']):>9} | {result_str:>8}")
 
    # ── summary ──────────────────────────────────────────────────────────────
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
    print("EVALUATION RESULTS — BC Policy")
    print(f"  Episodes       : {n}")
    print(f"  Grasped        : {n_grasped} / {n}  ({n_grasped/n*100:.1f}%)")
    print(f"  Released       : {n_released} / {n}  ({n_released/n*100:.1f}%)")
    print(f"  Full success   : {n_success} / {n}  ({n_success/n*100:.1f}%)")
    print(f"  Avg steps      : {avg_steps:.1f}  ({avg_steps*STEP_SLEEP:.2f}s)")
    print(f"  Avg grasp steps: {f'{np.mean(grasp_steps):.1f} ± {np.std(grasp_steps):.1f}' if grasp_steps else 'N/A'}")
    print(f"  Avg release steps:{f'{np.mean(release_steps):.1f} ± {np.std(release_steps):.1f}' if release_steps else 'N/A'}")
    print(f"  Avg place error: {f'{np.mean(place_errors):.4f} ± {np.std(place_errors):.4f}' if place_errors else 'N/A'}")
    print("\nFailure breakdown:")
    for k, v in failure_counts.items():
        print(f"  {k}: {v}")
    print("=" * 70)

    
    # ── save results ─────────────────────────────────────────────────────────
    out = {
        'policy':                'BC',
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
 
    os.makedirs(os.path.dirname(args.save_results), exist_ok=True)
    with open(args.save_results, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\nResults saved to: {args.save_results}")

#══════════════════════════════════════════════════════════════════════════════ # ENTRY POINT #══════════════════════════════════════════════════════════════════════════════
if __name__ == '__main__': 
    parser = argparse.ArgumentParser(description='BC policy evaluation in CoppeliaSim') 
    parser.add_argument('--checkpoint', type=str, 
                        default='./checkpoints/bc/bc_best.pth', help='Path to bc_best.pth') 
    parser.add_argument('--stats_dir', type=str, 
                        default='./checkpoints/bc/', help='Folder containing obs_mean/std and act_mean/std .npy files') 
    parser.add_argument('--n_episodes', type=int, 
                        default=N_EPISODES, help='Number of evaluation episodes') 
    parser.add_argument('--save_results', type=str, 
                        default='../results/bc_eval_results.json', help='Path to save JSON results (set to "" to skip)') 
    args = parser.parse_args() 
    run_evaluation(args)