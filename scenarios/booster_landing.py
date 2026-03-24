# scenarios/booster_landing.py
"""
Simulate New Shepard booster terminal descent and propulsive landing.
Real data: booster lands at ~2.7 m/s, ~3 km from launch site.
Entry conditions from public NS flight data.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
import matplotlib.pyplot as plt
from dynamics import Satellite6DOF

# Booster mass properties (public estimates)
# Dry mass ~20,000 kg, fuel remaining at landing ~1000 kg
env = Satellite6DOF(
    mass=21000.0,
    inertia=np.diag([1.2e6, 1.2e6, 5.0e4]),  # tall slender body
    dt=0.05
)

# Entry conditions: booster falling at ~250 m/s from ~5000m
state = env.reset(random_init=False)
env.state[0:3]  = np.array([0.0, 0.0, 5000.0])   # 5km altitude
env.state[3:6]  = np.array([0.0, 0.0, -250.0])    # falling at 250 m/s
env.state[6:10] = np.array([1.0, 0.0, 0.0, 0.0])  # upright
state = env.state.copy()

TARGET_LAND_VEL = 2.7   # m/s touchdown target
MAX_THRUST      = 490000.0  # BE-3PM ~490 kN throttleable

history = []

def landing_controller(state):
    """
    Proportional throttle to target touchdown velocity.
    Vertical axis only for this simplified scenario.
    """
    alt = state[2]
    vz  = state[5]
    g   = 9.81

    # Target velocity profile: linear from current to 2.7 m/s at ground
    v_target = -TARGET_LAND_VEL if alt < 100 else -max(2.7, min(250, alt * 0.05))

    # Throttle to match target velocity
    v_error  = v_target - vz
    thrust_z = env.mass * g + np.clip(500.0 * v_error, -MAX_THRUST * 0.8, MAX_THRUST * 0.8)
    thrust_z = np.clip(thrust_z, 0, MAX_THRUST)

    # Attitude hold (keep upright)
    q   = state[6:10]
    w   = state[10:13]
    Mxy = -5000.0 * np.array([q[1], q[2]]) - 1000.0 * w[:2]

    return np.array([0, 0, thrust_z, Mxy[0], Mxy[1], 0])

T = 4000
for t in range(T):
    action = landing_controller(state)
    state  = env.step(action)
    history.append({
        'time': t * env.dt,
        'alt':  state[2],
        'vz':   state[5],
        'thrust': action[2] / 1000,  # kN
    })
    if state[2] <= 0:
        print(f"Touchdown at t={t*env.dt:.1f}s | vz={state[5]:.2f} m/s "
              f"(target: {TARGET_LAND_VEL} m/s)")
        break

times   = [h['time']   for h in history]
alts    = [h['alt']    for h in history]
vzs     = [h['vz']     for h in history]
thrusts = [h['thrust'] for h in history]

fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
fig.suptitle('New Shepard Booster Propulsive Landing Simulation', fontweight='bold')

axes[0].plot(times, alts, color='steelblue')
axes[0].set_ylabel('Altitude (m)')
axes[0].axhline(0, color='red', linestyle='--', alpha=0.5)

axes[1].plot(times, vzs, color='darkorange')
axes[1].axhline(-TARGET_LAND_VEL, color='red', linestyle='--',
                label=f'Target: -{TARGET_LAND_VEL} m/s')
axes[1].set_ylabel('Vertical Velocity (m/s)')
axes[1].legend()

axes[2].plot(times, thrusts, color='purple')
axes[2].axhline(MAX_THRUST/1000, color='red', linestyle=':', alpha=0.5, label='Max thrust')
axes[2].set_ylabel('Thrust (kN)')
axes[2].set_xlabel('Time (s)')
axes[2].legend()

plt.tight_layout()
plt.savefig('booster_landing.png', dpi=150)
plt.show()