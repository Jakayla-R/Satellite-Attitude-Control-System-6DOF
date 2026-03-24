# scenarios/microgravity_hold.py
"""
Capsule attitude hold during NS-29 microgravity phase.
4 minutes of freefall above Karman line, capsule must maintain
stable orientation for payload operations before spin-up begins.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
import matplotlib.pyplot as plt
from dynamics import Satellite6DOF, quaternion_error_deg

MICROGRAVITY_DURATION = 240  # 4 minutes in seconds
dt = 0.01

env = Satellite6DOF(
    mass=6000.0,
    inertia=np.diag([12000.0, 12000.0, 8000.0]),
    dt=dt
)

# Start with slight attitude disturbance from separation event
state = env.reset(random_init=False)
env.state[6:10] = np.array([0.995, 0.07, 0.05, 0.02])
env.state[6:10] /= np.linalg.norm(env.state[6:10])
env.state[10:13] = np.array([0.02, -0.015, 0.008])
state = env.state.copy()

# Try loading trained DT model if available
model = None
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'model', 'checkpoints', 'dt_6dof.pt')
K = 20
if os.path.exists(MODEL_PATH):
    try:
        import torch
        from model.mingpt_6dof import DecisionTransformer6DOF
        model = DecisionTransformer6DOF(K=K)
        model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
        model.eval()
        print("Using trained Decision Transformer")
    except Exception as e:
        print(f"Falling back to PD: {e}")

states_buf  = np.zeros((K, 13), dtype=np.float32)
actions_buf = np.zeros((K, 6),  dtype=np.float32)
rtgs_buf    = np.full((K, 1), 5.0, dtype=np.float32)

from data.generate_trajectories import pd_controller, attitude_reward

history = []
T = int(MICROGRAVITY_DURATION / dt)

for t in range(T):
    states_buf[-1] = state

    if model is not None:
        from model.mingpt_6dof import DecisionTransformer6DOF
        action, unc = model.get_action(states_buf, actions_buf, rtgs_buf)
    else:
        action = pd_controller(state)
        unc    = np.zeros(6)

    state = env.step(action)
    ae    = quaternion_error_deg(state[6:10], np.array([1,0,0,0]))

    history.append({
        'time': t * dt,
        'att_error': ae,
        'uncertainty': float(np.mean(unc)),
    })

    states_buf  = np.roll(states_buf,  -1, axis=0)
    actions_buf = np.roll(actions_buf, -1, axis=0)
    rtgs_buf    = np.roll(rtgs_buf,    -1, axis=0)
    actions_buf[-1] = action
    rtgs_buf[-1]    = rtgs_buf[-2] - attitude_reward(state)

times  = [h['time']        for h in history]
errors = [h['att_error']   for h in history]
uncs   = [h['uncertainty'] for h in history]

fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
fig.suptitle('NS-29 Capsule Attitude Hold — 4 Min Microgravity Phase', fontweight='bold')

axes[0].plot(times, errors, color='steelblue')
axes[0].axhline(1.0, color='red', linestyle='--', alpha=0.7, label='1 deg threshold')
axes[0].axvline(240, color='green', linestyle='--', alpha=0.5, label='Spin-up begins')
axes[0].set_ylabel('Attitude Error (deg)')
axes[0].legend()

axes[1].plot(times, uncs, color='purple', alpha=0.8)
axes[1].set_ylabel('Controller Uncertainty')
axes[1].set_xlabel('Time (s)')

plt.tight_layout()
plt.savefig('microgravity_hold.png', dpi=150)
plt.show()

print(f"Final attitude error: {errors[-1]:.3f} deg")
print(f"Mean uncertainty:     {np.mean(uncs):.4f}")
ctrl = 'Decision Transformer' if model else 'PD baseline'
print(f"Controller used:      {ctrl}")