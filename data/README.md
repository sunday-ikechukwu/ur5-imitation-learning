# Dataset Documentation

This document describes the structure and schema of the demonstration dataset
collected for the UR5 pick-and-place imitation learning project.

---

## Overview

| Property | Value |
|---|---|
| Task | Pick and place (cube from pick table to drop zone) |
| Robot | UR5 with BarrettHand gripper |
| Simulator | CoppeliaSim (ZMQ Remote API) |
| Expert policy | Lua script running inside CoppeliaSim |
| Number of episodes | 100 |
| Simulation timestep | 50 ms |
| Intended polling frequency | 20 Hz |
| Effective dataset sampling rate | Variable (~2–3 Hz observed due to ZMQ polling overhead) |
| Timesteps per episode | ~85–95 (varies with cube start position) |
| File format | HDF5 (one file per episode) |

> [!NOTE]
> Data collection uses polling through the CoppeliaSim ZMQ Remote API.
> Although the simulator timestep is fixed at 50 ms (20 Hz),
> the effective recorded sampling frequency is lower and variable
> due to blocking remote API calls and communication overhead.
---

## File Naming

```
Pick_place_episode_{i+1}.hdf5     # i = 0 to 99
```

---

## HDF5 Schema

Each episode file contains the following datasets:

```bash
Example episode tree structure:

Pick_place_episode_1.hdf5
├── joint_positions        (T, 6)
├── end_effector_pose      (T, 7)
├── cube_position          (T, 3)
├── gripper_state          (T,)
├── actions                (T, 6)
├── timestamps             (T,)
└── attributes
    ├── task                   = "pick_and_place"
    ├── robot                  = "UR5"
    ├── gripper                = "BarrettHand"
    ├── num_joints             = 6
    ├── control_mode           = "joint_position"
    └── observation_space
```
---

### `joint_positions` — shape `(T, 6)`, dtype `float64`

Joint angles of the UR5 arm at observation time **t**, in radians.

| Index | Joint |
|---|---|
| 0 | `/UR5/joint[0]` — base rotation |
| 1 | `/UR5/joint[1]` — shoulder |
| 2 | `/UR5/joint[2]` — elbow |
| 3 | `/UR5/joint[3]` — wrist 1 |
| 4 | `/UR5/joint[4]` — wrist 2 |
| 5 | `/UR5/joint[5]` — wrist 3 |

---

### `actions` — shape `(T, 6)`, dtype `float64`

Target joint configuration executed by the expert policy at timestep `t+1`,
in radians. Same indexing as `joint_positions`.

This is the **action supervision signal** for behaviour cloning and ACT training.

> `actions[t]` is the joint configuration the arm transitions to after
> observing `joint_positions[t]`. The two arrays are offset by one timestep.

---
## Temporal Alignment

Observations and actions are temporally offset by one timestep:

- `joint_positions[t]` represents the robot state observed at time `t`
- `actions[t]` represents the target joint configuration executed immediately after observation

This produces standard `(state_t → action_t)` supervision for behavioural cloning.
---

### `end_effector_pose` — shape `(T, 7)`, dtype `float64`

Pose of the end-effector tip (`/UR5/BarrettHand/tip`) at each timestep.

| Index | Value |
|---|---|
| 0–2 | Position `(x, y, z)` in metres |
| 3–6 | Orientation as unit quaternion `(x, y, z, w)` |

> Coordinate frame: world frame 

---

### `cube_position` — shape `(T, 3)`, dtype `float64`

Position of the cube (`/Cube`) at each timestep, in metres.

| Index | Value |
|---|---|
| 0 | x |
| 1 | y |
| 2 | z |

- At episode start: z ≈ 0.38 m (resting on pick table)
- At episode end: z ≈ 0.30 m (placed at drop zone)

> Coordinate frame: world frame

---

### `gripper_state` — shape `(T,)`, dtype `float64`

Position of finger joint `/UR5/BarrettHand/jointC_2`, used as a proxy for
gripper open/closed state.

| Approximate value | State |
|---|---|
| ~0.786 | Open |
| ~1.14 | Closed (grasping) |

---

### `timestamps` — shape `(T,)`, dtype `float64`

Simulation time in seconds at each recorded sample.
Useful for synchronization, replay, latency analysis, and debugging timing issues.

---

## Observation and Action Spaces

```
Observation space (dim = 17):
  joint_positions      6
  end_effector_pose    7
  cube_position        3
  gripper_state        1
  ─────────────────────
  Total               17   (timestamps are not used as model inputs)

Action space (dim = 6):
  target joint angles  6
```
---
## File Attributes

The following metadata is stored as HDF5 file attributes:

| Attribute | Type | Description |
|---|---|---|
| `task` | `str` | Task name |
| `robot` | `str` | Robot platform |
| `gripper` | `str` | End-effector/gripper type |
| `num_joints` | `int` | Number of robot joints |
| `control_mode` | `str` | Robot control mode |
| `observation_space` | `str` | Recorded observation fields |
---

## Cube Position Randomisation

At the start of each episode the cube is placed at a uniformly sampled
position within the validated reachable workspace of the robot:

```
x  ~  Uniform(0.46,  1.1)   metres
y  ~  Uniform(-1.11, -0.76)  metres
z  =  0.38                  metres  (fixed — resting on table surface)
```

This range was manually validated to ensure the expert policy succeeds
reliably across the full region.

---

## Loading the Data

```python
import h5py
import numpy as np

with h5py.File('Pick_place_episode_1.hdf5', 'r') as f:
    joint_positions    = f['joint_positions'][:]     # (T, 6)
    actions            = f['actions'][:]             # (T, 6)
    end_effector_pose  = f['end_effector_pose'][:]   # (T, 7)
    cube_position      = f['cube_position'][:]       # (T, 3)
    gripper_state      = f['gripper_state'][:]       # (T,)

# Build observation vector at each timestep
observations = np.concatenate([
    joint_positions,       # 6
    end_effector_pose,     # 7
    cube_position,         # 3
    gripper_state[:, None] # 1
], axis=1)                 # → (T, 17)
```

---

## Dataset Validation & Future Improvements

Planned validation and preprocessing steps:

- [ ] Generate per-episode trajectory summary plots
- [ ] Generate dataset overview statistics
- [ ] Visualize cube trajectories in 3D
- [ ] Add movement-threshold filtering to remove frozen terminal frames
