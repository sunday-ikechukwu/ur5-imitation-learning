"""
record_trajectory.py
--------------------
Records joint trajectory for ONE episode for plotting purposes.

Usage:
    python record_trajectory.py --cube_x 0.744 --cube_y -0.863
"""

import os, sys, time, json, argparse, math
import numpy as np
import torch
import torch.nn as nn

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(PROJECT_ROOT)

from scripts.utils.zmq_remoteapi_connection_utils import connect_to_coppeliasim
from inference.act_inference import ACTPolicy, PositionalEncoding

# ── paste your ACTPolicy and PositionalEncoding classes here ──
# ── or import from act_inference.py if you've structured it as a module ──

OBS_DIM    = 17
ACTION_DIM = 6
CHUNK_SIZE = 10
MAX_STEPS  = 250

def get_handles(sim):
    cube_handle   = sim.getObject('/Cube')
    tip_handle    = sim.getObject('/UR5/BarrettHand/tip')
    joint_handles = [sim.getObject('/UR5/joint', {'index': i}) for i in range(6)]
    finger_joint  = sim.getObject('/UR5/BarrettHand/jointC_2')
    return cube_handle, tip_handle, joint_handles, finger_joint

def get_observation(sim, cube_handle, tip_handle, joint_handles, finger_joint):
    jp   = np.array([sim.getJointPosition(j) for j in joint_handles])
    ee   = np.array(sim.getObjectPose(tip_handle, -1))
    cube = np.array(sim.getObjectPosition(cube_handle, -1))
    grip = np.array([sim.getJointPosition(finger_joint)])
    return np.concatenate([jp, ee, cube, grip])

def predict_chunk(obs_np, model, obs_mean, obs_std, act_mean, act_std, device):
    obs_t = torch.tensor(obs_np, dtype=torch.float32).unsqueeze(0).to(device)
    obs_t = (obs_t - obs_mean) / obs_std
    with torch.no_grad():
        chunk_norm = model(obs_t)
    return (chunk_norm.squeeze(0) * act_std + act_mean).cpu().numpy()

def main(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # load policy
    model = ACTPolicy().to(device)
    model.eval()
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])

    obs_mean = torch.tensor(np.load(f'{args.stats_dir}/act_obs_mean.npy'), dtype=torch.float32).to(device)
    obs_std  = torch.tensor(np.load(f'{args.stats_dir}/act_obs_std.npy'),  dtype=torch.float32).to(device)
    act_mean = torch.tensor(np.load(f'{args.stats_dir}/act_act_mean.npy'), dtype=torch.float32).to(device)
    act_std  = torch.tensor(np.load(f'{args.stats_dir}/act_act_std.npy'),  dtype=torch.float32).to(device)

    sim = connect_to_coppeliasim()
    cube_handle, tip_handle, joint_handles, finger_joint = get_handles(sim)

    # set fixed cube position
    cube_pos = [args.cube_x, args.cube_y, 0.38]
    sim.setObjectPosition(cube_handle, -1, cube_pos)
    print(f'Cube set to: {cube_pos}')

    sim.clearInt32Signal('pick_and_place_done')
    sim.startSimulation()
    time.sleep(0.3)

    joint_trajectory = []
    chunk_buffer     = []
    step             = 0

    # ── add these ──
    grasped  = False
    released = False

    attach_handle = sim.getObject('/UR5/BarrettHand/attachPoint')
    place_handle  = sim.getObject('/placePose')
    hand_script   = sim.getObject('/UR5/BarrettHand/Script')
    place_pos     = np.array(sim.getObjectPosition(place_handle, -1))

    GRASP_XY_THRESHOLD = 0.06
    GRASP_Z_THRESHOLD  = 0.08
    PLACE_XY_THRESHOLD = 0.10
    PLACE_Z_THRESHOLD  = 0.10
    PLACE_TABLE_Z      = 0.30
    PLACE_Z_TOL        = 0.05
    SUCCESS_XY_THRESHOLD = 0.12
    # ───────────────

    while step < MAX_STEPS:
        if sim.getSimulationState() == sim.simulation_stopped:
            break

        obs = get_observation(sim, cube_handle, tip_handle,
                              joint_handles, finger_joint)
        ee_pos   = obs[6:9]
        cube_pos_obs = obs[13:16]

        joint_trajectory.append(obs[:6].tolist())

        if len(chunk_buffer) == 0:
            chunk        = predict_chunk(obs, model, obs_mean, obs_std,
                                         act_mean, act_std, device)
            chunk_buffer = list(chunk)

        action = chunk_buffer.pop(0)
        for j, jh in enumerate(joint_handles):
            sim.setJointTargetPosition(jh, float(action[j]))

        step += 1

        # ── grasp trigger ─────────────────────────────────────────────
        if not grasped:
            xy_dist = np.linalg.norm(ee_pos[:2] - cube_pos_obs[:2])
            z_dist  = abs(ee_pos[2] - cube_pos_obs[2])
            if xy_dist < GRASP_XY_THRESHOLD and z_dist < GRASP_Z_THRESHOLD:
                sim.setObjectParent(cube_handle, attach_handle, True)
                sim.callScriptFunction('closeHand', hand_script)
                if sim.getObjectParent(cube_handle) == attach_handle:
                    grasped = True
                    steps_to_grasp = step
                    chunk_buffer.clear()
                    time.sleep(0.5)

        # ── place trigger ──────────────────────────────────────────────
        elif grasped and not released:
            xy_dist = np.linalg.norm(ee_pos[:2] - place_pos[:2])
            z_dist  = abs(ee_pos[2] - place_pos[2])
            if xy_dist < PLACE_XY_THRESHOLD and z_dist < PLACE_Z_THRESHOLD:
                sim.setObjectParent(cube_handle, -1, True)
                sim.callScriptFunction('openHand', hand_script)
                released = True
                steps_to_release = step
                chunk_buffer.clear()
                time.sleep(0.5)

        # ── auto-stop once task complete ───────────────────────────────
        elif released:
            final_cube    = np.array(sim.getObjectPosition(cube_handle, -1))
            xy_dist_final = np.linalg.norm(final_cube[:2] - place_pos[:2])
            height_ok     = abs(final_cube[2] - PLACE_TABLE_Z) < PLACE_Z_TOL
            if xy_dist_final < SUCCESS_XY_THRESHOLD and height_ok:
                print(f'    [done] task complete at step {step}')
                break   # ← recording stops automatically here

    sim.stopSimulation()
    while sim.getSimulationState() != sim.simulation_stopped:
        time.sleep(0.1)

    # save trajectory
    out = {
        'policy':            'ACT',
        'cube_start':        cube_pos,
        'steps':             step,
        'steps_to_grasp':    steps_to_grasp,  
        'steps_to_release':  steps_to_release, 
        'joint_trajectory':  joint_trajectory,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.save_path)), exist_ok=True)
    with open(args.save_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f'Trajectory saved to: {args.save_path}  ({step} steps)')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str,
                        default='../checkpoints/act_checkpoints/act_best.pth')
    parser.add_argument('--stats_dir',  type=str,
                        default='../checkpoints/act_checkpoints/')
    parser.add_argument('--cube_x',     type=float, default=0.7443516268989547)
    parser.add_argument('--cube_y',     type=float, default=-0.8630068775851197)
    parser.add_argument('--save_path',  type=str,
                        default='../results/act_traj_episode.json')
    args = parser.parse_args()
    main(args)