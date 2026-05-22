"""
dataset_analysis.py
-------------------
Full dataset analysis for the UR5 pick-and-place imitation learning dataset.
Produces:
  - fig1_dataset_overview.png   : 6-panel overview card
  - fig2_trajectory_summary.png : joint + gripper trajectories for 6 episodes
  - fig3_cube_trajectories.png  : 3D cube trajectories + XY workspace coverage
  - dataset_stats.md            : raw statistics table for DATA.md / README

Usage:
  python dataset_analysis.py --data_dir /path/to/hdf5/folder --out_dir ./assets
"""

import argparse
import glob
import os

import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# ── colour palette ─────────────────────────────────────────────────────────────
BLUE   = '#2563EB'
ORANGE = '#F97316'
GREEN  = '#16A34A'
PURPLE = '#7C3AED'
RED    = '#DC2626'
TEAL   = '#0D9488'
GRAY   = '#6B7280'
JCOLS  = [BLUE, ORANGE, GREEN, PURPLE, RED, TEAL]

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 10,
    'axes.spines.top': False,
    'axes.spines.right': False,
})


# ══════════════════════════════════════════════════════════════════════════════
# 1. DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════

def load_all_episodes(data_dir: str):
    """Load all HDF5 episode files and return a list of dicts."""
    pattern = os.path.join(data_dir, "Pick_place_episode_*.hdf5")
    files = sorted(
        glob.glob(pattern),
        key=lambda x: int(os.path.basename(x).split("_")[-1].replace(".hdf5", ""))
    )
    if not files:
        raise FileNotFoundError(f"No HDF5 files found at: {pattern}")
    print(f"Found {len(files)} episodes.")

    episodes = []
    for fpath in files:
        ep_idx = int(os.path.basename(fpath).split("_")[-1].replace(".hdf5", ""))
        with h5py.File(fpath, 'r') as f:
            episodes.append({
                'idx':  ep_idx,
                'jp':   f['joint_positions'][:],       # (T, 6)
                'ac':   f['actions'][:],               # (T, 6)
                'cube': f['cube_position'][:],         # (T, 3)
                'ee':   f['end_effector_pose'][:],     # (T, 7)
                'grip': f['gripper_state'][:],         # (T,)
                'ts':   f['timestamps'][:],   # (T,)
            })
    return episodes


# ══════════════════════════════════════════════════════════════════════════════
# 2. STATISTICS
# ══════════════════════════════════════════════════════════════════════════════

def compute_stats(episodes):
    """Compute per-episode and aggregate statistics."""
    per_ep = []
    jp_global_min = np.full(6,  np.inf)
    jp_global_max = np.full(6, -np.inf)

    for ep in episodes:
        jp   = ep['jp']
        ac   = ep['ac']
        cube = ep['cube']
        grip = ep['grip']
        ts   = ep['ts']
        T = jp.shape[0]

        # real duration from simulation timestamps
        duration_s = float(ts[-1] - ts[0])

        # timestep statistics
        dt = np.diff(ts)

        mean_dt = float(dt.mean())
        std_dt  = float(dt.std())

        effective_hz = float(1.0 / mean_dt)

        # frozen frames: timesteps where NO joint moves more than 0.1 mrad
        diff = np.abs(np.diff(jp, axis=0))
        frozen_count = int((diff.max(axis=1) < 1e-4).sum())
        frozen_pct   = frozen_count / max(T - 1, 1) * 100

        # gripper open/close transitions
        closed       = (grip > 0.95).astype(int)
        transitions  = int(np.abs(np.diff(closed)).sum())

        # mean action step size (how much the arm moves each timestep)
        ac_delta = float(np.abs(ac - jp).mean())

        jp_global_min = np.minimum(jp_global_min, jp.min(axis=0))
        jp_global_max = np.maximum(jp_global_max, jp.max(axis=0))

        per_ep.append({
            'idx':          ep['idx'],
            'T':            T,
            'duration_s':   duration_s,
            'mean_dt':      mean_dt,
            'std_dt':       std_dt,
            'effective_hz': effective_hz,
            'frozen_count': frozen_count,
            'frozen_pct':   frozen_pct,
            'transitions':  transitions,
            'ac_delta':     ac_delta,
            'cube_x0':      float(cube[0, 0]),
            'cube_y0':      float(cube[0, 1]),
            'cube_zf':      float(cube[-1, 2]),
        })

    T_all       = np.array([e['T']           for e in per_ep])
    dur_all     = np.array([e['duration_s']  for e in per_ep])
    dt_all      = np.array([e['mean_dt']     for e in per_ep])
    std_dt_all  = np.array([e['std_dt']      for e in per_ep])
    hz_all      = np.array([e['effective_hz'] for e in per_ep])
    frozen_all  = np.array([e['frozen_pct']  for e in per_ep])
    delta_all   = np.array([e['ac_delta']    for e in per_ep])
    cube_x0     = np.array([e['cube_x0']     for e in per_ep])
    cube_y0     = np.array([e['cube_y0']     for e in per_ep])
    trans_all   = np.array([e['transitions'] for e in per_ep])

    agg = {
        # episode length
        'T_mean': T_all.mean(),   'T_std': T_all.std(),
        'T_min':  T_all.min(),    'T_max': T_all.max(),
        'T_total': int(T_all.sum()),

        # duration
        'dur_mean': dur_all.mean(),
        'dur_std': dur_all.std(),
        'dur_total': dur_all.sum(),

        # timing statistics
        'dt_mean': dt_all.mean(),
        'dt_std': std_dt_all.mean(),

        'hz_mean': hz_all.mean(),
        'hz_std': hz_all.std(),

        # frozen frames
        'frozen_mean': frozen_all.mean(), 'frozen_max': frozen_all.max(),
        'frozen_episodes_gt5pct': int((frozen_all > 5).sum()),

        # workspace coverage
        'cube_x_min': cube_x0.min(), 'cube_x_max': cube_x0.max(), 'cube_x_mean': cube_x0.mean(),
        'cube_y_min': cube_y0.min(), 'cube_y_max': cube_y0.max(), 'cube_y_mean': cube_y0.mean(),

        # joint ranges
        'jp_global_min': jp_global_min,
        'jp_global_max': jp_global_max,
        'jp_ranges_deg': np.degrees(jp_global_max - jp_global_min),

        # action delta
        'delta_mean': delta_all.mean() * 1000,   # mrad
        'delta_min':  delta_all.min()  * 1000,
        'delta_max':  delta_all.max()  * 1000,

        # gripper
        'trans_mean': trans_all.mean(),
        'all_complete': bool((trans_all >= 2).all()),

        # episode count
        'n_episodes': len(per_ep),
    }

    return per_ep, agg


def print_stats(agg):
    """Print a readable summary to stdout."""
    print("\n" + "="*60)
    print("DATASET STATISTICS")
    print("="*60)

    print(f"\nEpisode count        : {agg['n_episodes']}")

    print(f"\n── Timing ─────────────────────────────────")
    print(f"  simulator timestep : 50 ms (20 Hz nominal)")
    print(f"  mean timestep      : {agg['dt_mean']:.3f} ± {agg['dt_std']:.3f} s")
    print(f"  effective rate     : {agg['hz_mean']:.2f} ± {agg['hz_std']:.2f} Hz")

    print(f"\n── Episode Length ──────────────────────────")
    print(f"  mean ± std         : {agg['T_mean']:.1f} ± {agg['T_std']:.1f} timesteps")
    print(f"  min / max          : {agg['T_min']} / {agg['T_max']} timesteps")
    print(f"  total timesteps    : {agg['T_total']:,}")

    print(f"\n── Duration ────────────────────────────────")
    print(f"  mean ± std         : {agg['dur_mean']:.2f} ± {agg['dur_std']:.2f} s")
    print(f"  total              : {agg['dur_total']:.1f} s  ({agg['dur_total']/60:.1f} min)")

    print(f"\n── Frozen Frames ───────────────────────────")
    print(f"  mean per episode   : {agg['frozen_mean']:.2f}%")
    print(f"  max in any episode : {agg['frozen_max']:.2f}%")
    print(f"  episodes > 5%      : {agg['frozen_episodes_gt5pct']}")

    print(f"\n── Cube Workspace Coverage ─────────────────")
    print(f"  x range            : [{agg['cube_x_min']:.4f}, {agg['cube_x_max']:.4f}] m  (mean {agg['cube_x_mean']:.4f})")
    print(f"  y range            : [{agg['cube_y_min']:.4f}, {agg['cube_y_max']:.4f}] m  (mean {agg['cube_y_mean']:.4f})")

    print(f"\n── Joint-Angle Ranges (across all episodes) ─")
    for j in range(6):
        lo = agg['jp_global_min'][j]; hi = agg['jp_global_max'][j]
        print(f"  Joint {j}  : [{np.degrees(lo):+7.2f}°, {np.degrees(hi):+7.2f}°]"
              f"  range = {agg['jp_ranges_deg'][j]:.1f}°")

    print(f"\n── Action Step Size ────────────────────────")
    print(f"  mean |action−obs|  : {agg['delta_mean']:.1f} mrad/step")
    print(f"  min / max          : {agg['delta_min']:.1f} / {agg['delta_max']:.1f} mrad/step")

    print(f"\n── Gripper ─────────────────────────────────")
    print(f"  mean transitions   : {agg['trans_mean']:.2f} / episode")
    print(f"  all episodes complete (≥2 transitions): {agg['all_complete']}")
    print()


def save_stats_md(agg, out_dir: str):
    """Write dataset_stats.md — paste into DATA.md or README."""
    lines = [
        "## Dataset Statistics\n",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Episodes | {agg['n_episodes']} |",
        f"| Simulator timestep | 50 ms (20 Hz nominal) |",
        f"| Effective dataset sampling rate | {agg['hz_mean']:.2f} ± {agg['hz_std']:.2f} Hz |",
        f"| Mean timestep interval | {agg['dt_mean']:.3f} ± {agg['dt_std']:.3f} s |",
        f"| Mean episode length | {agg['T_mean']:.1f} ± {agg['T_std']:.1f} timesteps |",
        f"| Min / max episode length | {agg['T_min']} / {agg['T_max']} timesteps |",
        f"| Total timesteps | {agg['T_total']:,} |",
        f"| Mean episode duration | {agg['dur_mean']:.2f} ± {agg['dur_std']:.2f} s |",
        f"| Total recording time | {agg['dur_total']:.1f} s ({agg['dur_total']/60:.1f} min) |",
        f"| Mean frozen-frame % | {agg['frozen_mean']:.2f}% |",
        f"| Cube x coverage | [{agg['cube_x_min']:.3f}, {agg['cube_x_max']:.3f}] m |",
        f"| Cube y coverage | [{agg['cube_y_min']:.3f}, {agg['cube_y_max']:.3f}] m |",
        f"| Mean action step size | {agg['delta_mean']:.1f} mrad / step |",
        f"| Gripper transitions / episode | {agg['trans_mean']:.1f} (open → close → open) |",
        f"| All episodes complete | {'Yes' if agg['all_complete'] else 'No'} |",
        "",
        "### Joint Range of Motion\n",
        "| Joint | Min (°) | Max (°) | Range (°) |",
        "|-------|---------|---------|-----------|",
    ]
    for j in range(6):
        lo  = np.degrees(agg['jp_global_min'][j])
        hi  = np.degrees(agg['jp_global_max'][j])
        rng = agg['jp_ranges_deg'][j]
        lines.append(f"| Joint {j} | {lo:+.1f} | {hi:+.1f} | {rng:.1f} |")

    md_path = os.path.join(out_dir, "dataset_stats.md")
    with open(md_path, 'w', encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Saved: {md_path}")


# ══════════════════════════════════════════════════════════════════════════════
# 3. FIGURE 1 — Dataset Overview Card
# ══════════════════════════════════════════════════════════════════════════════

def plot_overview(episodes, per_ep, agg, out_dir: str):
    T_all      = np.array([e['T']          for e in per_ep])
    frozen_all = np.array([e['frozen_pct'] for e in per_ep])
    delta_all  = np.array([e['ac_delta']   for e in per_ep]) * 1000   # mrad
    cube_x0    = np.array([e['cube_x0']    for e in per_ep])
    cube_y0    = np.array([e['cube_y0']    for e in per_ep])

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle(
        "UR5 Pick-and-Place — Dataset Overview  (100 Demonstrations)",
        fontsize=14, fontweight='bold', y=1.01
    )

    # 1a — episode length histogram
    ax = axes[0, 0]
    ax.hist(T_all, bins=15, color=BLUE, edgecolor='white', linewidth=0.6)
    ax.axvline(T_all.mean(), color=RED, lw=1.8, linestyle='--',
               label=f'mean = {T_all.mean():.1f}')
    ax.set_title("Episode Length (timesteps)", fontweight='bold')
    ax.set_xlabel("Timesteps"); ax.set_ylabel("Count")
    ax.legend(fontsize=9)

    # 1b — effective duration histogram
    ax = axes[0, 1]
    dur = np.array([e['duration_s'] for e in per_ep])
    ax.hist(dur, bins=15, color=TEAL, edgecolor='white', linewidth=0.6)
    ax.axvline(dur.mean(), color=RED, lw=1.8, linestyle='--',
               label=f'mean = {dur.mean():.2f} s')
    ax.set_title("Recorded Episode Duration (seconds)", fontweight='bold')
    ax.set_xlabel("Duration (s)"); ax.set_ylabel("Count")
    ax.legend(fontsize=9)

    # 1c — frozen frame % per episode
    ax = axes[0, 2]
    ax.bar(range(1, len(per_ep) + 1), frozen_all, color=ORANGE, width=0.8)
    ax.axhline(frozen_all.mean(), color=RED, lw=1.5, linestyle='--',
               label=f'mean = {frozen_all.mean():.1f}%')
    ax.set_title("Frozen-Frame % per Episode", fontweight='bold')
    ax.set_xlabel("Episode"); ax.set_ylabel("Frozen frames (%)")
    ax.legend(fontsize=9)

    # 1d — cube workspace scatter
    ax = axes[1, 0]
    sc = ax.scatter(cube_x0, cube_y0, c=range(len(per_ep)),
                    cmap='viridis', s=40, alpha=0.85)
    rect = Rectangle(
        (0.46, -1.11),           # bottom-left corner
        1.10 - 0.46,             # width
        -0.76 - (-1.11),         # height
        linewidth=1.5,
        edgecolor=RED,
        facecolor='none',
        linestyle='--',
        label='validated workspace'
    )
    ax.add_patch(rect)
    ax.set_title("Cube Start Position Coverage", fontweight='bold')
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    ax.legend(fontsize=8)
    plt.colorbar(sc, ax=ax, label='Episode').ax.tick_params(labelsize=8)

    # 1e — joint range of motion
    ax = axes[1, 1]
    bars = ax.bar(range(6), agg['jp_ranges_deg'], color=JCOLS, edgecolor='white')
    ax.set_title("Joint Range of Motion (°)", fontweight='bold')
    ax.set_xlabel("Joint index"); ax.set_ylabel("Range (degrees)")
    ax.set_xticks(range(6))
    ax.set_xticklabels([f'J{i}' for i in range(6)])
    for bar, v in zip(bars, agg['jp_ranges_deg']):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f'{v:.0f}°', ha='center', va='bottom', fontsize=8)

    # 1f — action step size distribution
    ax = axes[1, 2]
    ax.hist(delta_all, bins=20, color=PURPLE, edgecolor='white', linewidth=0.6)
    ax.axvline(delta_all.mean(), color=RED, lw=1.8, linestyle='--',
               label=f'mean = {delta_all.mean():.1f} mrad')
    ax.set_title("Mean Action Step Size", fontweight='bold')
    ax.set_xlabel("Mean |action − obs| (mrad)"); ax.set_ylabel("Count")
    ax.legend(fontsize=9)

    fig.tight_layout()
    out = os.path.join(out_dir, "fig1_dataset_overview.png")
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {out}")


# ══════════════════════════════════════════════════════════════════════════════
# 4. FIGURE 2 — Trajectory Summaries
# ══════════════════════════════════════════════════════════════════════════════

def plot_trajectories(episodes, out_dir: str):
    n = len(episodes)
    # pick 6 evenly spaced episodes
    sample_ids = [0, n//5, 2*n//5, 3*n//5, 4*n//5, n-1]

    fig, axes = plt.subplots(6, 2, figsize=(14, 18))
    fig.suptitle("Timestamp-Aligned Trajectory Summaries — 6 Representative Episodes",
                 fontsize=13, fontweight='bold', y=1.005)

    for row, idx in enumerate(sample_ids):
        ep   = episodes[idx]

        jp   = ep['jp']
        grip = ep['grip']
        ts   = ep['ts']

        T = jp.shape[0]

        # normalize timestamps so episode starts at t=0
        t = ts - ts[0]

        ax_j = axes[row, 0]
        for j in range(6):
            ax_j.plot(t, np.degrees(jp[:, j]), color=JCOLS[j], lw=1.3, label=f'J{j}')
        ax_j.set_title(f"Episode {ep['idx']} — Joint Angles",
                       fontsize=9, fontweight='bold')
        ax_j.set_ylabel("Angle (°)", fontsize=8)
        ax_j.tick_params(labelsize=7)
        if row == 0:
            ax_j.legend(fontsize=6, ncol=6, loc='upper right')
        if row == 5:
            ax_j.set_xlabel("Time (s)", fontsize=8)

        ax_g = axes[row, 1]
        ax_g.plot(t, grip, color=ORANGE, lw=1.5)
        ax_g.fill_between(t, grip.min(), grip, alpha=0.15, color=ORANGE)
        ax_g.axhline(0.95, color=GRAY, lw=0.8, linestyle=':',
                     label='close threshold')
        ax_g.set_title(f"Episode {ep['idx']} — Gripper State",
                       fontsize=9, fontweight='bold')
        ax_g.set_ylabel("Finger joint position", fontsize=8)
        ax_g.tick_params(labelsize=7)
        if row == 0:
            ax_g.legend(fontsize=7)
        if row == 5:
            ax_g.set_xlabel("Time (s)", fontsize=8)

    fig.tight_layout()
    out = os.path.join(out_dir, "fig2_trajectory_summary.png")
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {out}")


# ══════════════════════════════════════════════════════════════════════════════
# 5. FIGURE 3 — Cube 3D Trajectories
# ══════════════════════════════════════════════════════════════════════════════

def plot_cube_trajectories(episodes, per_ep, out_dir: str):
    cube_x0 = np.array([e['cube_x0'] for e in per_ep])
    cube_y0 = np.array([e['cube_y0'] for e in per_ep])
    n = len(episodes)

    fig = plt.figure(figsize=(13, 6))
    fig.suptitle("Cube 3D Trajectories — All Episodes",
                 fontsize=13, fontweight='bold')

    # 3D trajectories
    ax3d = fig.add_subplot(121, projection='3d')
    for idx, ep in enumerate(episodes):
        cube  = ep['cube']
        color = plt.cm.coolwarm(idx / n)
        ax3d.plot(cube[:, 0], cube[:, 1], cube[:, 2],
                  color=color, lw=0.7, alpha=0.25)
        ax3d.scatter(cube[0, 0],  cube[0, 1],  cube[0, 2],
                     color='green', s=8, alpha=0.5, zorder=3)
        ax3d.scatter(cube[-1, 0], cube[-1, 1], cube[-1, 2],
                     color='red',   s=8, alpha=0.5, zorder=3)
    ax3d.set_xlabel("x (m)", fontsize=8); ax3d.set_ylabel("y (m)", fontsize=8)
    ax3d.set_zlabel("z (m)", fontsize=8)
    ax3d.set_title("3D view  (green=start, red=end)", fontsize=9)
    ax3d.tick_params(labelsize=7)

    # XY workspace scatter
    ax2d = fig.add_subplot(122)
    sc = ax2d.scatter(cube_x0, cube_y0, c=range(n),
                      cmap='viridis', s=45, alpha=0.9,
                      edgecolors='white', linewidths=0.3)
    rect = Rectangle((0.46, -1.11), 0.64 , 0.35,
                        linewidth=1.5,
                        edgecolor=RED,
                        facecolor='none',
                        linestyle='--',
                        label='validated workspace'
                    )
    ax2d.add_patch(rect)
    ax2d.set_title("Cube Start Position Coverage (XY plane)", fontsize=9, fontweight='bold')
    ax2d.set_xlabel("x (m)"); ax2d.set_ylabel("y (m)")
    ax2d.legend(fontsize=8)
    plt.colorbar(sc, ax=ax2d, label='Episode index').ax.tick_params(labelsize=8)

    fig.tight_layout()
    out = os.path.join(out_dir, "fig3_cube_trajectories.png")
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {out}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="UR5 dataset analysis")
    parser.add_argument('--data_dir', type=str,
                        default=r"C:\Users\PRECIOUS WEAL\robot_learning_pick_place\pick_and_place_imitation_data",
                        help="Folder containing Pick_place_episode_*.hdf5 files")
    parser.add_argument('--out_dir', type=str, default="./assets",
                        help="Output folder for figures and stats markdown")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print(f"\nLoading episodes from: {args.data_dir}")
    episodes = load_all_episodes(args.data_dir)

    print("Computing statistics...")
    per_ep, agg = compute_stats(episodes)

    print_stats(agg)
    save_stats_md(agg, args.out_dir)

    print("\nGenerating figures...")
    plot_overview(episodes, per_ep, agg, args.out_dir)
    plot_trajectories(episodes, args.out_dir)
    plot_cube_trajectories(episodes, per_ep, args.out_dir)

    print("\nDone. All outputs saved to:", args.out_dir)


if __name__ == "__main__":
    main()