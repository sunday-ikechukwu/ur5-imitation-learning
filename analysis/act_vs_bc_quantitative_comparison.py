import json
import matplotlib.pyplot as plt
import numpy as np
import os 
m = None

with open('../results/bc_eval_results.json', 'r')  as f:
    bc_results = json.load(f)

with open('../results/act_eval_results.json', 'r') as f:
    act_results = json.load(f)

# Extract metrics
metrics = ["n_episodes" ,'n_grasped', 'n_released', 'n_success', "success_rate", "grasp_rate", "release_rate", "completion_rate", "avg_steps", "avg_steps_to_grasp", "avg_steps_to_release", "avg_placement_error_xy", "failure_breakdown"]

metric_names = {
    "n_episodes": "Episodes",
    "n_grasped": "Grasped Episodes",
    "n_released": "Released Episodes",
    "n_success": "Successful Episodes",

    "success_rate": "Success Rate (%)",
    "grasp_rate": "Grasp Rate (%)",
    "release_rate": "Release Rate (%)",
    "completion_rate": "Completion Rate (%)",

    "avg_steps": "Avg Total Steps",
    "avg_steps_to_grasp": "Avg Steps to Grasp",
    "avg_steps_to_release": "Avg Steps to Release",

    "avg_placement_error_xy": "Avg Placement Error XY (m)",

    "failure_breakdown": "Failure Breakdown"
}

bc_metrics = {m: bc_results.get(m, None)  for m in metrics}
act_metrics = {m: act_results.get(m, None)  for m in metrics}
 
def format_value(v):
    if v is None:
        return "—"
    
    if isinstance(v, dict):
        if len(v) == 0:
            return 'None'
        return "\n".join(f"{k}: {format_value(val)}" for k, val in v.items())

    if isinstance(v, float):
        return f"{v:.2f}"
    return str(v)

# Build comparison table
def build_comparison_table():
    columns = ['Metric', 'BC', 'ACT']
    rows = [
        [
            metric_names[m], 
            format_value(bc_metrics[m]), 
            format_value(act_metrics[m])
        ]
        for m in metrics
    ]

    fig, ax = plt.subplots(figsize=(12, len(rows)*0.5 + 1))
    ax.axis('off')
    ax.set_title("Quantitative Comparison: BC vs ACT", fontsize=16, fontweight='bold', pad = 1)

    table = ax.table(
        cellText=rows,
        colLabels=columns,
        cellLoc='center',
        loc='center',
        bbox=[0, 0.05, 1, 0.9]
    )
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 1.5)

    # Style header row
    for j in range(len(columns)):
        table[0, j].set_facecolor('#2c3e50')
        table[0, j].set_text_props(color='white', fontweight='bold')

    # Style alternating rows
    for i in range(1, len(rows) + 1):
        for j in range(len(columns)):
            if i % 2 == 0:
                table[i, j].set_facecolor('#f2f2f2')

    plt.tight_layout()

    save_path = '../assets/comparison_plots/act_vs_bc_quantitative_comparison_table.png'
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight')
    plt.show()

POLICY_COLORS = {
    'BC': '#E66100',   # Muted Orange
    'ACT': '#5D3A9B'   # Muted Purple
}

def failure_breakdown_plot(axes_target):
    policies = ['BC', 'ACT']
    total_episodes = 20
    
    # Raw counts
    successes = np.array([0, 20])
    stuck_failures = np.array([20, 0])

    # Normalize to percentages for a 100% stacked bar
    success_pct = (successes / total_episodes) * 100
    stuck_pct = (stuck_failures / total_episodes) * 100
    
    SUCCESS_COLOR = '#44AA99' # Muted Teal/Green
    STUCK_COLOR = '#AA4499'   # Muted Magenta/

    # Plotting the stacked bars
    bars_success = axes_target.bar(policies, success_pct, color=SUCCESS_COLOR, 
                                   label='Success', width=0.3, edgecolor='white')
    
    bars_stuck = axes_target.bar(policies, stuck_pct, bottom=success_pct,   color=STUCK_COLOR, 
                                 label='Failed: Stuck', width=0.3, edgecolor='white')
    
    # --- MODERNIZATION ---
    axes_target.set_ylim(0, 105)
    axes_target.spines['top'].set_visible(False)
    axes_target.spines['right'].set_visible(False)
    
    axes_target.yaxis.grid(True, linestyle='--', alpha=0.6, color='gray')
    axes_target.set_axisbelow(True) 

    axes_target.set_ylabel('Proportion of Episodes (%)', fontsize=12, fontweight='medium')
    axes_target.set_title("Episode Outcome Breakdown", fontweight='bold')

    # Adding a sleek legend inside the plot area
    axes_target.legend(loc='upper center', frameon=False, fontsize=10)

def success_rate_bar_plot():
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle(
        "BC vs. ACT Performance Evaluation",
        fontsize=14, fontweight='bold', y= 1
    )

    #____________________________________________________________________
    # success rate plot
    #_____________________________________________________________________
    ax = axes[0,0]
    policies = ['BC', 'ACT']
    success_rate = [bc_metrics['success_rate'], act_metrics['success_rate']]

    bar_colors = [POLICY_COLORS[policy] for policy in policies]
    bars=ax.bar(policies, success_rate, color=bar_colors, width=0.3)
    
    ax.set_ylim(0, 115)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.yaxis.grid(True, linestyle='--', alpha=0.6, color='gray')
    ax.set_axisbelow(True)

    ax.set_ylabel('Success Rate (%)', fontsize=12, fontweight='medium')
    ax.set_title("Success Rate Comparison", fontweight='bold')

    labels = [f"{v:.1f}%" for v in success_rate]
    ax.bar_label(bars, labels=labels, padding=5, fontsize=11, fontweight='bold', color='#333333')

    #_____________________________________________________________________
    #       AVERAGE STEP
    #____________________________________________________________________
    ax = axes[0,1]
    policies = ['BC', 'ACT']
    avg_steps = [bc_metrics['avg_steps'], act_metrics['avg_steps']]

    bar_colors = [POLICY_COLORS[policy] for policy in policies]
    bars=ax.bar(policies, avg_steps, color=bar_colors, width=0.3)
    
    ax.set_ylim()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.yaxis.grid(True, linestyle='--', alpha=0.6, color='gray')
    ax.set_axisbelow(True)

    ax.set_ylabel('Avg Total Steps', fontsize=12, fontweight='medium')
    ax.set_title("Task Completion Efficiency",fontweight='bold')

    labels = [v for v in avg_steps]
    ax.bar_label(bars, labels=labels, padding=1, fontsize=11, fontweight='bold', color='#333333')

    #__________________________________________________________________
    # PLACEMENT ERROR COMPARISON
    #__________________________________________________________________
    ax = axes[0,2]
    failure_breakdown_plot(ax)

    #__________________________________________________________________
    # PLACEMENT ERROR COMPARISON
    #__________________________________________________________________
    ax = axes[1,0]
    policies = ['BC', 'ACT']
    raw_bc_error = bc_metrics.get('avg_placement_error_xy')
    raw_act_error = act_metrics.get('avg_placement_error_xy')
    raw_errors = [raw_bc_error, raw_act_error]

    # Replace 'None' with 0 specifically for drawing the bars
    plot_errors = [0 if v is None else v for v in raw_errors]


    bar_colors = [POLICY_COLORS[policy] for policy in policies]
    bars=ax.bar(policies, plot_errors, color=bar_colors, width=0.3)
    
    max_error = max([e for e in plot_errors if e is not None] + [0.01])
    ax.set_ylim(0, max_error * 1.2) # Give it 20% headroom for the labels

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.yaxis.grid(True, linestyle='--', alpha=0.6, color='gray')
    ax.set_axisbelow(True)

    ax.set_ylabel('Avg Placement Error XY (m)', fontsize=12, fontweight='medium')
    ax.set_title("Average Placement Error Comparison", fontweight='bold')

    labels = ["N/A" if v is None else f"{v:.4f}m" for v in raw_errors]
    ax.bar_label(bars, labels=labels, padding=1, fontsize=11, fontweight='bold', color='#333333')

    #__________________________________________________________________
    # scatter of final cube positions
    #__________________________________________________________________
    cube_finals = np.array([[ep['cube_final'][0], ep['cube_final'][1]]
                         for ep in act_results['episodes']])
    place_pos   = np.array([act_results['episodes'][0]['place_pos'][0],
                         act_results['episodes'][0]['place_pos'][1]])
    errors      = np.array([ep['placement_error_xy'] for ep in 
                            act_results['episodes']])
    episodes    = [ep['episode'] for ep in act_results['episodes']]

    ax = axes[1,1]
    sc = ax.scatter(cube_finals[:, 0], cube_finals[:, 1],
                     c=errors, cmap='RdYlGn_r', s=80, zorder=3)
    ax.scatter(*place_pos, marker='*', s=300,
                color='blue', label='Target', zorder=4)
    circle = plt.Circle(place_pos, 0.12, color='blue',
                    fill=False, linestyle='--', label='Success threshold (12cm)')
    ax.add_patch(circle)
    plt.colorbar(sc, ax=ax, label='XY Error (m)')
    ax.set_title('Cube Final Positions vs Target', fontweight='bold')
    ax.set_xlabel('x (m)'); ax.set_ylabel('y (m)')
    ax.legend(fontsize=8); ax.set_aspect('equal')

    #__________________________________________________________________
    # placement error per episode
    #__________________________________________________________________
    ax = axes[1,2]
    ax.bar(episodes, errors * 100, color='#2563EB', edgecolor='white')
    ax.axhline(np.mean(errors) * 100, color='red', linestyle='--',
                    label=f'Mean = {np.mean(errors)*100:.1f} cm')
    ax.axhline(12, color='gray', linestyle=':', alpha=0.7,
                    label='Success threshold (12 cm)')
    ax.set_title('Placement Error per Episode', fontweight='bold')

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.set_ylim(0,13)
    ax.set_xlabel('Episode'); ax.set_ylabel('XY Error (cm)')
    ax.legend(fontsize=8)

    save_path = '../assets/comparison_plots/act_vs_bc_comparison_plots.png'
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight')

    fig.tight_layout(pad=3.0, h_pad=4.0, w_pad=3.0, rect=[0, 0, 1, 0.97])
    plt.subplots_adjust(top=0.9)
    plt.show()


    

def main():
    build_comparison_table()
    success_rate_bar_plot()
    

if __name__ == '__main__':
    main()