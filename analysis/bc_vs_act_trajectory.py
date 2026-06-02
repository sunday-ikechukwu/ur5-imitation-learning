import numpy as np
import matplotlib.pyplot as plt
import json

# load trajectory data
with open('../results/bc_traj_episode.json') as f:
    bc_data = json.load(f)
with open('../results/act_traj_episode.json') as f:
    act_data = json.load(f)

bc_traj = np.array(bc_data['joint_trajectory'])
act_traj = np.array(act_data['joint_trajectory'])

 # ACT grasp and release markers
act_grasp   = act_data.get('steps_to_grasp')
act_release = act_data.get('steps_to_release')
# BC grasp
bc_grasp    = bc_data.get('steps_to_grasp')

JCOLS = ['#2563EB','#F97316','#16A34A','#7C3AED','#DC2626','#0D9488']

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
fig.suptitle('Joint Trajectories: BC vs ACT — Same Cube Start Position',
             fontsize=13, fontweight='bold')

for j, ax in enumerate(axes.flat):
    t_bc  = np.arange(len(bc_traj))
    t_act = np.arange(len(act_traj))

    ax.plot(t_bc,  np.degrees(bc_traj[:,  j]),
            color=JCOLS[j], lw=1.5, linestyle='--',
            alpha=0.7, label='BC')
    ax.plot(t_act, np.degrees(act_traj[:, j]),
            color=JCOLS[j], lw=2.0, linestyle='-',
            label='ACT')

    # shade the BC stuck region — from where it stops moving
    STUCK_WINDOW = 20
    bc_rolling = np.array([
        np.abs(bc_traj[min(t + STUCK_WINDOW, len(bc_traj)-1), j] - bc_traj[t, j])
        for t in range(len(bc_traj))
    ])
    stuck_from = np.where(bc_rolling < 0.01)[0] 
    if len(stuck_from) > 0:
        first_stuck = stuck_from[0]
        shade_start = max(first_stuck, (bc_grasp + 5) if bc_grasp else 30)
        ax.axvspan(shade_start, len(bc_traj), alpha=0.08, color='red',
                label='BC stuck' if j == 0 else None)

    if act_grasp is not None:
        ax.axvline(act_grasp, color='green', linestyle='--',
                lw=1.5, label='ACT grasp' if j == 0 else None)
    if act_release is not None:
        ax.axvline(act_release, color='red', linestyle='--',
                lw=1.5, label='ACT release' if j == 0 else None)
    
    # BC grasp marker
    if bc_grasp is not None:
        ax.axvline(bc_grasp, color='green', linestyle=':',
                   lw=1.5, alpha=0.7, label='BC grasp' if j == 0 else None)

    ax.set_title(f'Joint {j}', fontsize=9, fontweight='bold')
    ax.set_xlabel('Timestep', fontsize=8)
    ax.set_ylabel('Angle (°)', fontsize=8)
    ax.tick_params(labelsize=7)
    if j == 0:
        ax.legend(fontsize=7)

plt.tight_layout()
plt.savefig('../assets/comparison_plots/bc_vs_act_joint_trajectory.png', dpi=150, bbox_inches='tight')
plt.show()