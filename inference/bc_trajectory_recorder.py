"""
record_trajectory.py
--------------------
Records joint trajectory for ONE episode for plotting purposes.

Usage:
    python record_trajectory.py --cube_x 0.744 --cube_y -0.863
"""

import os, sys, time, json, argparse
import numpy as np
import torch
import torch.nn as nn

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(PROJECT_ROOT)

from scripts.utils.zmq_remoteapi_connection_utils import connect_to_coppeliasim
from inference.bc_inference import BCPolicy

OBS_DIM    = 17
ACTION_DIM = 6
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

def attach_cube(sim, cube_handle, attach_handle, hand_script):
    sim.setObjectParent(cube_handle, attach_handle, True)
    sim.callScriptFunction('closeHand', hand_script)

def detach_cube(sim, cube_handle, hand_script):
    sim.setObjectParent(cube_handle, -1, True)
    sim.callScriptFunction('openHand', hand_script)

def predict_action(obs_np, model, obs_mean, obs_std, act_mean, act_std, device):
    """Normalise obs → run policy → denormalise → return joint targets (6,)."""
    obs_t = torch.tensor(obs_np, dtype=torch.float32).unsqueeze(0).to(device)
    obs_t = (obs_t - obs_mean) / (obs_std + 1e-8) 
    
    with torch.no_grad(): 
        action_norm = model(obs_t) # (1, 6) normalised 
    action = (action_norm * act_std + act_mean).squeeze(0).cpu().numpy() # (6,) 
    return action 

def main(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # load policy
    model = BCPolicy().to(device)
    model.eval()
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])

    obs_mean = torch.tensor(np.load(f'{args.stats_dir}/obs_mean.npy'),dtype=torch.float32).to(device) 
    obs_std  = torch.tensor(np.load(f'{args.stats_dir}/obs_std.npy'), dtype=torch.float32).to(device) 
    act_mean = torch.tensor(np.load(f'{args.stats_dir}/act_mean.npy'), dtype=torch.float32).to(device) 
    act_std  = torch.tensor(np.load(f'{args.stats_dir}/act_std.npy'), dtype=torch.float32).to(device) 

    sim = connect_to_coppeliasim()
    cube_handle, tip_handle, joint_handles, finger_joint = get_handles(sim)

    # set fixed cube position
    cube_start = [args.cube_x, args.cube_y, 0.38]
    sim.setObjectPosition(cube_handle, -1, cube_start)
    print(f'Cube set to: {cube_start}')

    sim.clearInt32Signal('pick_and_place_done')
    sim.startSimulation()
    time.sleep(0.3)

    joint_trajectory = []
    step             = 0

    grasped  = False
    released = False
    steps_to_grasp   = None
    steps_to_release = None
    cube_lifted      = False
    final_cube       = np.array([args.cube_x, args.cube_y, 0.38])

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

    STUCK_WINDOW    = 50 # number of steps to look back when checking if stuck
    STUCK_THRESHOLD = 0.005

    ee_history  = []
    # ───────────────

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
        joint_trajectory.append(obs[:6].tolist())
 
        # ── predict & act ─────────────────────────────────────────────────
        action = predict_action(obs, model, obs_mean, obs_std,
                                act_mean, act_std, device)
        for j, jh in enumerate(joint_handles):
            sim.setJointTargetPosition(jh, float(action[j])) #sim.setJointTargetPosition
 
        time.sleep(0.05)

        # ── stuck detection ───────────────────────────────────────────────
        ee_history.append(ee_pos.copy())
        if len(ee_history) > STUCK_WINDOW:
            movement = np.linalg.norm(
                ee_history[-1] - ee_history[-STUCK_WINDOW]
            )
            if movement < STUCK_THRESHOLD:
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

    # save trajectory
    out = {
        'policy':           'BC',
        'cube_start':        cube_start,
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
                        default='../checkpoints/bc_checkpoints/bc_best.pth')
    parser.add_argument('--stats_dir',  type=str,
                        default='../checkpoints/bc_checkpoints/')
    parser.add_argument('--cube_x',     type=float, default=0.7443516268989547)
    parser.add_argument('--cube_y',     type=float, default=-0.8630068775851197)
    parser.add_argument('--save_path',  type=str,
                        default='../results/bc_traj_episode.json')
    args = parser.parse_args()
    main(args)