"""
6DOF Satellite Dynamics Engine
Extends the 3DOF pitch-only Simulink baseline to full rigid body.

Original parameters preserved:
  - Izz = 10 kg*m^2 (pitch axis, from your Simulink model)
  - dt  = 0.01s
  - Euler integration (mirrors rt_OneStep in ert_main.c)
"""

import numpy as np

STATE_DIM  = 13   # [x, y, z, vx, vy, vz, q0, q1, q2, q3, wx, wy, wz]
ACTION_DIM = 6    # [Fx, Fy, Fz, Mx, My, Mz]


def quaternion_multiply(q1, q2):
    """Hamilton product of two quaternions (scalar-first convention)."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ])


def quaternion_error(q, q_ref):
    """Attitude error quaternion: q_err = q_ref_conj * q"""
    q_ref_conj = np.array([q_ref[0], -q_ref[1], -q_ref[2], -q_ref[3]])
    return quaternion_multiply(q_ref_conj, q)


def quaternion_error_deg(q, q_ref):
    """Geodesic attitude error in degrees."""
    dot = abs(np.dot(q, q_ref))
    dot = np.clip(dot, 0.0, 1.0)
    return np.degrees(2 * np.arccos(dot))


class Satellite6DOF:
    """
    Full 6DOF rigid body satellite dynamics.

    State:   [x, y, z, vx, vy, vz, q0, q1, q2, q3, wx, wy, wz]  (13,)
    Control: [Fx, Fy, Fz, Mx, My, Mz]                             (6,)

    Inertia tensor diagonal preserves your original Izz=10 (pitch axis = z).
    """

    def __init__(self, mass=500.0, inertia=None, dt=0.01):
        self.mass = mass
        self.dt   = dt
        # Diagonal inertia: Ixx, Iyy, Izz
        # Izz=10 matches your Simulink model's moment of inertia
        self.I     = inertia if inertia is not None else np.diag([8.5, 9.0, 10.0])
        self.I_inv = np.linalg.inv(self.I)
        self.state = self._zero_state()

    def _zero_state(self):
        return np.array([
            0.0, 0.0, 0.0,       # position
            0.0, 0.0, 0.0,       # velocity
            1.0, 0.0, 0.0, 0.0,  # quaternion (identity)
            0.0, 0.0, 0.0        # angular velocity
        ])

    def reset(self, random_init=False, seed=None):
        if seed is not None:
            np.random.seed(seed)

        self.state = self._zero_state()

        if random_init:
            self.state[0:3]  = np.random.uniform(-10, 10, 3)   # position
            self.state[3:6]  = np.random.uniform(-1,  1,  3)   # velocity
            self.state[10:13]= np.random.uniform(-0.1, 0.1, 3) # angular rate
            # small random attitude perturbation
            axis  = np.random.randn(3)
            axis /= np.linalg.norm(axis)
            angle = np.random.uniform(0, np.radians(30))
            self.state[6:10] = np.array([
                np.cos(angle/2),
                *(np.sin(angle/2) * axis)
            ])

        return self.state.copy()

    def step(self, control, noise_std=0.0):
        """
        Euler integration step.
        Mirrors rt_OneStep() from your ert_main.c.
        """
        F = control[:3]
        M = control[3:]

        pos = self.state[0:3]
        vel = self.state[3:6]
        q   = self.state[6:10]
        w   = self.state[10:13]

        # --- Translational dynamics ---
        a       = F / self.mass
        pos_new = pos + vel * self.dt
        vel_new = vel + a   * self.dt

        # --- Rotational dynamics (Euler's equation) ---
        w_dot = self.I_inv @ (M - np.cross(w, self.I @ w))
        w_new = w + w_dot * self.dt

        # --- Quaternion kinematics ---
        wx, wy, wz = w
        Omega = 0.5 * np.array([
            [0,   -wx, -wy, -wz],
            [wx,   0,   wz, -wy],
            [wy,  -wz,  0,   wx],
            [wz,   wy, -wx,  0 ]
        ])
        q_new = q + Omega @ q * self.dt
        q_new /= np.linalg.norm(q_new)  # keep unit quaternion

        # --- Optional process noise (for uncertainty training data) ---
        if noise_std > 0.0:
            pos_new += np.random.normal(0, noise_std, 3)
            vel_new += np.random.normal(0, noise_std * 0.1, 3)
            w_new   += np.random.normal(0, noise_std * 0.01, 3)

        self.state = np.concatenate([pos_new, vel_new, q_new, w_new])
        return self.state.copy()

    @property
    def pos(self):  return self.state[0:3]

    @property
    def vel(self):  return self.state[3:6]

    @property
    def quat(self): return self.state[6:10]

    @property
    def omega(self): return self.state[10:13]
