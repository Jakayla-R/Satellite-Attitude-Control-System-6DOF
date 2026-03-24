"""
Visualization for 6DOF satellite GNC evaluation.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def plot_attitude_error(metrics, save_path=None):
    fig, axes = plt.subplots(4, 1, figsize=(11, 10), sharex=True)
    fig.suptitle("6DOF Decision Transformer: GNC Performance", fontsize=13, fontweight='bold')

    axes[0].plot(metrics["att_errors"], color="steelblue", linewidth=1.2)
    axes[0].axhline(1.0, color="red", linestyle="--", alpha=0.7, label="1 deg threshold")
    axes[0].set_ylabel("Att Error (deg)")
    axes[0].legend(fontsize=8)

    axes[1].plot(metrics["pos_errors"], color="darkorange", linewidth=1.2)
    axes[1].set_ylabel("Pos Error (m)")

    axes[2].plot(metrics["uncertainties"], color="purple", alpha=0.8, linewidth=1.0)
    # Shade flagged steps
    flagged = np.array(metrics["flagged_steps"])
    axes[2].fill_between(range(len(flagged)), 0, flagged * max(metrics["uncertainties"]),
                         alpha=0.2, color="red", label="High uncertainty")
    axes[2].set_ylabel("Action Uncertainty")
    axes[2].legend(fontsize=8)

    axes[3].plot(metrics["omega_norms"], color="teal", linewidth=1.0)
    axes[3].set_ylabel("Omega Norm (rad/s)")
    axes[3].set_xlabel("Timestep")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.show()


def plot_trajectory_3d(history, save_path=None):
    positions = np.array([h["state"][0:3] for h in history])

    fig = plt.figure(figsize=(8, 7))
    ax  = fig.add_subplot(111, projection='3d')

    sc = ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2],
                    c=np.arange(len(positions)), cmap='viridis',
                    s=2, alpha=0.6)
    ax.scatter(*positions[0],  color='green', s=60, zorder=5, label='Start')
    ax.scatter(*positions[-1], color='red',   s=60, zorder=5, label='End')
    ax.scatter(0, 0, 0, color='black', marker='*', s=100, zorder=5, label='Target')

    plt.colorbar(sc, ax=ax, label='Timestep', shrink=0.6)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title("Satellite Trajectory (6DOF)")
    ax.legend(fontsize=8)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.show()


def plot_quaternion_components(history, save_path=None):
    quats  = np.array([h["state"][6:10] for h in history])
    labels = ["q0 (scalar)", "q1", "q2", "q3"]
    colors = ["black", "steelblue", "darkorange", "green"]

    fig, ax = plt.subplots(figsize=(11, 4))
    for i in range(4):
        ax.plot(quats[:, i], label=labels[i], color=colors[i], linewidth=1.0)
    ax.axhline(1.0, color='black', linestyle=':', alpha=0.2)
    ax.axhline(0.0, color='black', linestyle=':', alpha=0.2)
    ax.set_ylabel("Quaternion Component")
    ax.set_xlabel("Timestep")
    ax.set_title("Attitude Quaternion History (target: q0=1, q1=q2=q3=0)")
    ax.legend(ncol=4, fontsize=9)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.show()


def plot_action_history(history, save_path=None):
    actions = np.array([h["action"] for h in history])
    labels  = ["Fx", "Fy", "Fz", "Mx", "My", "Mz"]
    colors  = ["#e41a1c","#377eb8","#4daf4a","#984ea3","#ff7f00","#a65628"]

    fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    fig.suptitle("Control Actions over Episode", fontsize=12)

    for i in range(3):
        axes[0].plot(actions[:, i], label=labels[i], color=colors[i], linewidth=0.9)
    axes[0].set_ylabel("Force (N)")
    axes[0].legend(ncol=3, fontsize=8)

    for i in range(3, 6):
        axes[1].plot(actions[:, i], label=labels[i], color=colors[i], linewidth=0.9)
    axes[1].set_ylabel("Torque (N*m)")
    axes[1].set_xlabel("Timestep")
    axes[1].legend(ncol=3, fontsize=8)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.show()
