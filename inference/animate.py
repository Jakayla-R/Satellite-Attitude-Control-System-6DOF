"""
Live 3D animation of the 6DOF satellite using your trained Decision Transformer.

Run: python inference/animate.py

Controls:
  - Click and drag to rotate the 3D view
  - Close the window to stop
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from dynamics import Satellite6DOF
from data.generate_trajectories import attitude_reward, TARGET_STATE

MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'model', 'checkpoints', 'dt_6dof.pt')
K          = 20
TARGET_RTG = 5.0
TRAIL_LEN  = 80


def load_model():
    import torch
    from model.mingpt_6dof import DecisionTransformer6DOF
    m = DecisionTransformer6DOF(K=K)
    m.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
    m.eval()
    return m


def pd_fallback(state, Kp_att=5.0, Kd_att=1.0, Kp_pos=1.0, Kd_pos=0.5):
    """PD controller used if no trained model is found."""
    from dynamics import quaternion_error
    q     = state[6:10]
    w     = state[10:13]
    q_ref = TARGET_STATE[6:10]
    q_err = quaternion_error(q, q_ref)
    torque = -Kp_att * q_err[1:4] - Kd_att * w
    force  = (-Kp_pos * state[0:3] - Kd_pos * state[3:6])
    return np.concatenate([force, torque]), np.zeros(6)


def quat_to_rotation_matrix(q):
    w, x, y, z = q
    return np.array([
        [1-2*(y*y+z*z),   2*(x*y-w*z),   2*(x*z+w*y)],
        [  2*(x*y+w*z), 1-2*(x*x+z*z),   2*(y*z-w*x)],
        [  2*(x*z-w*y),   2*(y*z+w*x), 1-2*(x*x+y*y)]
    ])


def make_satellite_geometry(pos, q, scale=0.5):
    """Returns body faces and solar panel faces for plotting."""
    R = quat_to_rotation_matrix(q)
    p = np.array(pos)
    s = scale

    # Body cube corners (local frame)
    c = np.array([[-1,-1,-1],[ 1,-1,-1],[ 1, 1,-1],[-1, 1,-1],
                  [-1,-1, 1],[ 1,-1, 1],[ 1, 1, 1],[-1, 1, 1]]) * s

    # Rotate and translate
    cw = (R @ c.T).T + p

    body_faces = [
        [cw[0],cw[1],cw[2],cw[3]],
        [cw[4],cw[5],cw[6],cw[7]],
        [cw[0],cw[1],cw[5],cw[4]],
        [cw[2],cw[3],cw[7],cw[6]],
        [cw[1],cw[2],cw[6],cw[5]],
        [cw[0],cw[3],cw[7],cw[4]],
    ]

    # Solar panels (along local x-axis)
    pw = 1.2 * s
    ph = 0.35 * s
    panel_corners_L = np.array([[-1-2*pw,-ph,-0.05],[-1,-ph,-0.05],
                                  [-1, ph,-0.05],[-1-2*pw, ph,-0.05]]) * s
    panel_corners_R = np.array([[ 1,     -ph,-0.05],[ 1+2*pw,-ph,-0.05],
                                  [ 1+2*pw, ph,-0.05],[ 1,      ph,-0.05]]) * s

    plw = (R @ panel_corners_L.T).T + p
    prw = (R @ panel_corners_R.T).T + p

    panel_faces = [[plw[0],plw[1],plw[2],plw[3]],
                   [prw[0],prw[1],prw[2],prw[3]]]

    # Body z-axis arrow tip
    z_tip = p + R @ np.array([0, 0, 1.4*s])

    return body_faces, panel_faces, z_tip


def run_animation(use_model=True):
    # Try loading model
    model = None
    if use_model and os.path.exists(MODEL_PATH):
        try:
            model = load_model()
            print("Loaded trained Decision Transformer.")
        except Exception as e:
            print(f"Could not load model ({e}), falling back to PD controller.")
    else:
        print("No checkpoint found, using PD controller baseline.")

    # Sim state
    env   = Satellite6DOF()
    state = env.reset(random_init=True, seed=7)

    # Context buffers for DT
    states_buf  = np.zeros((K, 13), dtype=np.float32)
    actions_buf = np.zeros((K, 6),  dtype=np.float32)
    rtgs_buf    = np.full((K, 1), TARGET_RTG, dtype=np.float32)

    trail = []
    att_history  = []
    pos_history  = []
    unc_history  = []
    step_count   = [0]

    # ---- Figure setup ----
    fig = plt.figure(figsize=(13, 6), facecolor='#0e0e12')
    fig.suptitle('6DOF Autonomous GNC — Decision Transformer', color='white',
                 fontsize=12, fontweight='bold', y=0.98)

    ax3d = fig.add_axes([0.0, 0.05, 0.55, 0.90], projection='3d')
    ax3d.set_facecolor('#0e0e12')
    ax3d.xaxis.pane.fill = False
    ax3d.yaxis.pane.fill = False
    ax3d.zaxis.pane.fill = False
    ax3d.xaxis.pane.set_edgecolor('#ffffff15')
    ax3d.yaxis.pane.set_edgecolor('#ffffff15')
    ax3d.zaxis.pane.set_edgecolor('#ffffff15')
    ax3d.tick_params(colors='#ffffff50', labelsize=7)
    ax3d.set_xlabel('X', color='#ff6b6b', labelpad=2, fontsize=8)
    ax3d.set_ylabel('Y', color='#6bff9e', labelpad=2, fontsize=8)
    ax3d.set_zlabel('Z', color='#6bb5ff', labelpad=2, fontsize=8)

    # Right panel: 3 metric plots
    ax_att = fig.add_axes([0.58, 0.68, 0.40, 0.25], facecolor='#0e0e12')
    ax_pos = fig.add_axes([0.58, 0.38, 0.40, 0.25], facecolor='#0e0e12')
    ax_unc = fig.add_axes([0.58, 0.08, 0.40, 0.25], facecolor='#0e0e12')

    for ax in [ax_att, ax_pos, ax_unc]:
        ax.tick_params(colors='#ffffff60', labelsize=7)
        for spine in ax.spines.values():
            spine.set_edgecolor('#ffffff20')

    ax_att.set_ylabel('Att error (deg)', color='#ffffff80', fontsize=8)
    ax_pos.set_ylabel('Pos error (m)',   color='#ffffff80', fontsize=8)
    ax_unc.set_ylabel('Uncertainty',     color='#ffffff80', fontsize=8)
    ax_unc.set_xlabel('Timestep',        color='#ffffff80', fontsize=8)

    line_att, = ax_att.plot([], [], color='#ff6b6b', linewidth=1)
    line_pos, = ax_pos.plot([], [], color='#6bb5ff', linewidth=1)
    line_unc, = ax_unc.plot([], [], color='#c49bff', linewidth=1, alpha=0.8)
    ax_att.axhline(1.0, color='#ff6b6b', linestyle='--', alpha=0.4, linewidth=0.8)

    # Text overlays
    info_text = ax3d.text2D(0.02, 0.97, '', transform=ax3d.transAxes,
                             color='white', fontsize=8, va='top',
                             fontfamily='monospace')
    ctrl_text = ax3d.text2D(0.02, 0.06, '', transform=ax3d.transAxes,
                             color='#c49bff', fontsize=8, va='bottom',
                             fontfamily='monospace')

    # Drawable objects
    trail_line,   = ax3d.plot([], [], [], color='#8b5cf6', alpha=0.5, linewidth=1)
    target_dot,   = ax3d.plot([0],[0],[0], 'o', color='#ff6b6b', markersize=5, alpha=0.7)
    z_arrow_line, = ax3d.plot([], [], [], color='#fbbf24', linewidth=2)

    body_poly  = Poly3DCollection([], alpha=0.45, linewidth=0.5)
    panel_poly = Poly3DCollection([], alpha=0.6,  linewidth=0.5)
    ax3d.add_collection3d(body_poly)
    ax3d.add_collection3d(panel_poly)

    STEPS_PER_FRAME = 3

    def update(frame):
        nonlocal state

        for _ in range(STEPS_PER_FRAME):
            states_buf[-1] = state

            if model is not None:
                action, uncertainty = model.get_action(states_buf, actions_buf, rtgs_buf)
            else:
                action, uncertainty = pd_fallback(state)

            next_state = env.step(action)

            # Shift context buffers
            states_buf[:-1]  = states_buf[1:]
            actions_buf[:-1] = actions_buf[1:]
            rtgs_buf[:-1]    = rtgs_buf[1:]
            actions_buf[-1]  = action
            rtgs_buf[-1]     = rtgs_buf[-2] - attitude_reward(state)

            trail.append(state[0:3].copy())
            if len(trail) > TRAIL_LEN:
                trail.pop(0)

            # Metrics
            q     = state[6:10]
            dot   = abs(np.dot(q, np.array([1,0,0,0])))
            ae    = np.degrees(2 * np.arccos(np.clip(dot, 0, 1)))
            pe    = np.linalg.norm(state[0:3])
            unc   = float(np.mean(uncertainty))

            att_history.append(ae)
            pos_history.append(pe)
            unc_history.append(unc)
            step_count[0] += 1
            state = next_state

        # ---- Update 3D view ----
        pos3 = state[0:3]
        q3   = state[6:10]

        body_faces, panel_faces, z_tip = make_satellite_geometry(pos3, q3)
        body_poly.set_verts(body_faces)
        body_poly.set_facecolor('#1D6FA8')
        body_poly.set_edgecolor('#5ab4ff')
        panel_poly.set_verts(panel_faces)
        panel_poly.set_facecolor('#1D9E75')
        panel_poly.set_edgecolor('#5affd3')

        z_arrow_line.set_data_3d([pos3[0], z_tip[0]],
                                  [pos3[1], z_tip[1]],
                                  [pos3[2], z_tip[2]])

        if len(trail) > 1:
            tr = np.array(trail)
            trail_line.set_data_3d(tr[:,0], tr[:,1], tr[:,2])

        # Auto-scale view around satellite
        r = max(3.0, np.linalg.norm(pos3) * 1.5)
        ax3d.set_xlim(pos3[0]-r, pos3[0]+r)
        ax3d.set_ylim(pos3[1]-r, pos3[1]+r)
        ax3d.set_zlim(pos3[2]-r, pos3[2]+r)

        # Info overlay
        ae_now  = att_history[-1]
        pe_now  = pos_history[-1]
        unc_now = unc_history[-1]
        ctrl    = 'Decision Transformer' if model else 'PD baseline'
        info_text.set_text(
            f't={step_count[0]:5d}   att={ae_now:6.2f} deg   '
            f'pos={pe_now:5.2f} m   unc={unc_now:.3f}'
        )
        ctrl_text.set_text(f'controller: {ctrl}')

        # ---- Update metric plots ----
        xs = list(range(len(att_history)))
        line_att.set_data(xs, att_history)
        line_pos.set_data(xs, pos_history)
        line_unc.set_data(xs, unc_history)

        for ax, hist in [(ax_att, att_history), (ax_pos, pos_history), (ax_unc, unc_history)]:
            ax.set_xlim(0, max(len(hist), 100))
            mn, mx = min(hist), max(hist)
            pad = max((mx - mn) * 0.1, 0.1)
            ax.set_ylim(mn - pad, mx + pad)

        return body_poly, panel_poly, z_arrow_line, trail_line, \
               line_att, line_pos, line_unc, info_text, ctrl_text

    ani = animation.FuncAnimation(
        fig, update, interval=30, blit=False, cache_frame_data=False
    )

    plt.show()


if __name__ == '__main__':
    run_animation(use_model=True)
