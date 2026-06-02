"""
pick_and_place_imitation_data_recorder_utils.py
Records (observation, action) pairs with UR5 lua script as the expert policy.
Saves to HDF5 for offline behavior cloning training.
"""
import numpy as np
import h5py
from pathlib import Path
import time

# path to save the recorded data
save_path = Path(r"C:\Users\PRECIOUS WEAL\robot_learning_pick_place\pick_and_place_imitation_data")

save_path.mkdir(parents=True, exist_ok=True)  # Create directory if it doesn't exist

#========================HANDLES====================
def get_handles(sim):
    '''Get the handles for the observations and action parameters'''
    cube_handle = sim.getObject('/Cube')
    tip_handle = sim.getObject('/UR5/BarrettHand/tip')

    joint_handles = []
    for i in range(6):
        joint_handles.append(sim.getObject(f'/UR5/joint', {'index': i}))

    # Finger joint handles
    finger_joint = sim.getObject('/UR5/BarrettHand/jointC_2')

    attach_handle = sim.getObject('/UR5/BarrettHand/attachPoint')
    place_handle  = sim.getObject('/placePose')
    hand_script    = sim.getObject('/UR5/BarrettHand/Script')

    return (
        cube_handle,
        tip_handle,
        joint_handles,
        finger_joint,
        attach_handle,
        place_handle,
        hand_script
    )

def record_episode(sim, episode_index):
    """
    Poll sensors every 50 ms until the pick-and-place signal fires.
    All sim.* calls stay on the calling thread (ZMQ is not thread-safe).
    Saves collected data to HDF5 before returning.
    """
    
    cube_handle, tip_handle, joint_handles, finger_joint = get_handles(sim)

    #===============Buffers to store data during recording======================
    joint_positions_buffer = []           # current arm configuration
    cube_position_buffer = []             # where the cube is
    end_effector_pose_buffer = []     # current gripper/tip pose
    gripper_state_buffer = []             # open or closed
    action_buffer = []                   # target joint positions
    time_buffer = []

    previous_joint_positions = None # previous timestep state for observation-action pairing

    print(f"Episode {episode_index}: recording...")
    
    while (sim.getSimulationState() != sim.simulation_stopped and sim.getInt32Signal('pick_and_place_done') != 1):

        # read sensor values every time step
        current_joint_positions = [sim.getJointPosition(joint) for joint in joint_handles]

        cube_position = sim.getObjectPosition(cube_handle, -1)
        ee_pose = sim.getObjectPose(tip_handle, -1)
        gripper_state = sim.getJointPosition(finger_joint)  # Assuming jointC_2 controls the gripper opening/closing

         # Action: only record once we have a previous position to diff against
        if previous_joint_positions is not None:
            sim_time = sim.getSimulationTime()
            joint_positions_buffer.append(previous_joint_positions)
            cube_position_buffer.append(cube_position)
            end_effector_pose_buffer.append(ee_pose)
            gripper_state_buffer.append(gripper_state)
            action_buffer.append(current_joint_positions) # action at time t
            time_buffer.append(sim_time)

        previous_joint_positions = current_joint_positions.copy()

        time.sleep(0.05)
    _save_episode(joint_positions_buffer, end_effector_pose_buffer, cube_position_buffer, gripper_state_buffer, action_buffer, time_buffer, episode_index)


def _save_episode(joint_positions_buffer, end_effector_pose_buffer, cube_position_buffer, gripper_state_buffer, action_buffer, time_buffer, episode_index):
    '''Save the recorded episode data to an HDF5 file. Only saves up to the minimum length of the buffers to ensure data alignment.'''

    min_len = min(
        len(joint_positions_buffer), 
        len(end_effector_pose_buffer), 
        len(cube_position_buffer), 
        len(gripper_state_buffer), 
        len(action_buffer),
        len(time_buffer)
    )

    if min_len == 0:
        print(f"Episode {episode_index}: no data recorded, skipping save.")
        return
    
    file_path = save_path / f"Pick_place_episode_{episode_index}.hdf5"

    with h5py.File(file_path, 'w') as f:
        # ================= METADATA =================
        f.attrs['task'] = 'pick_and_place'
        f.attrs['robot'] = 'UR5'
        f.attrs['gripper'] = 'BarrettHand'
        f.attrs['num_joints'] = 6
        f.attrs['control_mode'] = 'joint_position'
        f.attrs['observation_space'] = (
            'joint_positions, cube_position, '
            'end_effector_pose, gripper_state'
        )

        # ================= DATASETS =================
        f.create_dataset('joint_positions',      data=np.array(joint_positions_buffer[:min_len]))
        f.create_dataset('end_effector_pose',   data=np.array(end_effector_pose_buffer[:min_len]))
        f.create_dataset('cube_position', data=np.array(cube_position_buffer[:min_len]))
        f.create_dataset('gripper_state', data=np.array(gripper_state_buffer[:min_len]))
        f.create_dataset('actions',    data=np.array(action_buffer[:min_len]))
        f.create_dataset('time_step',     data=np.array(time_buffer[:min_len]))

    print(f"Episode {episode_index}: {min_len} timesteps saved to {file_path}")
