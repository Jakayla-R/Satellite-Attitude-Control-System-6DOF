"""
Generate offline training trajectories using a baseline PD controller.
Saves to data/trajectories.npy

Run: python data/generate_trajectories.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from dynamics import Satellite6DOF, quaternion_error

TARGET_STATE = np.array([0,0,0, 0,0,0, 1,0,0,0, 0,0,0], dtype=np.float32)


def pd_controller(state, target=TARGET_STATE, Kp_att=5.0, Kd_att=1.0,
                  Kp_pos=1.0, Kd_pos=0.5):
    """
    PD controller generalized from your original Simulink PID.
    Attitude error via quaternion product, position error via Euclidean norm.
    """
    q     = state[6:10]
    w     = state[10:13]
    q_ref = target[6:10]

    q_err  = quaternion_error(q, q_ref)
    torque = -Kp_att * q_err[1:4] - Kd_att * w

    force  = (-Kp_pos * (state[0:3]  - target[0:3])
              -Kd_pos * (state[3:6]  - target[3:6]))

    return np.concatenate([force, torque])


def attitude_reward(state, target=TARGET_STATE):
    """Negative geodesic attitude error (maximized at perfect alignment)."""
    q     = state[6:10]
    q_ref = target[6:10]
    dot   = abs(np.dot(q, q_ref))
    dot   = np.clip(dot, 0.0, 1.0)
    att_err = 2 * np.arccos(dot)
    pos_err = np.linalg.norm(state[0:3] - target[0:3])
    return -(att_err + 0.1 * pos_err)


def generate_trajectories(n_traj=500, T=200, noise_std=0.01, seed=42):
    """
    Returns array of shape (n_traj, T, 19)
    Columns: [state(13), action(6)]
    """
    np.random.seed(seed)
    env  = Satellite6DOF(dt=0.01)
    data = np.zeros((n_traj, T, 19), dtype=np.float32)

    for i in range(n_traj):
        state = env.reset(random_init=True)
        for t in range(T):
            action  = pd_controller(state)
            action += np.random.normal(0, noise_std, 6)
            data[i, t, :13] = state
            data[i, t, 13:] = action
            state = env.step(action, noise_std=noise_std)

        if (i + 1) % 100 == 0:
            print(f"  Generated {i+1}/{n_traj} trajectories")

    return data


if __name__ == "__main__":
    print("Generating trajectories...")
    data = generate_trajectories(n_traj=500, T=500)
    out_path = os.path.join(os.path.dirname(__file__), "trajectories.npy")
    np.save(out_path, data)
    print(f"Saved {data.shape} -> {out_path}")
