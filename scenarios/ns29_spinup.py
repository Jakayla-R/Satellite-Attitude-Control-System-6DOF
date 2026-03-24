# scenarios/ns29_spinup.py
"""
Simulate NS-29 capsule spin-up to 11 RPM using your 6DOF dynamics.
Real mission data: Blue Origin NS-29, Feb 4 2025
Capsule spun to ~1.15 rad/s (11 RPM) around z-axis for lunar-g simulation.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
import matplotlib.pyplot as plt
from dynamics import Satellite6DOF

TARGET_RPM  = 11.0
TARGET_RADS = TARGET_RPM * 2 * np.pi / 60  # 1.152 rad/s

# NS-29 capsule mass properties (estimated from public data)
# Capsule diameter ~3.7m, mass ~6000 kg
# Izz dominates for spin axis
env = Satellite6DOF(
    mass=6000.0,
    inertia=np.diag([12000.0, 12000.0, 8000.0]),  # kg*m^2, z = spin axis
    dt=0.1
)

state = env.reset(random_init=False)
history = []

# RCS torque authority (estimated ~500 N*m per thruster pair)
RCS_TORQUE = 300.0

def spinup_controller(state):
    """
    Spin up around z-axis to target rate.
    Pure angular rate tracking, no attitude hold.
    """
    wz = state[12]  # z angular rate
    error = TARGET_RADS - wz
    # Proportional with saturation
    Mz = np.clip(50.0 * error, -RCS_TORQUE, RCS_TORQUE)
    return np.array([0, 0, 0, 0, 0, Mz])

T = 3000  # 300 seconds of sim
for t in range(T):
    action = spinup_controller(state)
    state  = env.step(action)
    rpm    = state[12] * 60 / (2 * np.pi)
    history.append({
        'time': t * env.dt,
        'rpm':  rpm,
        'wz':   state[12],
        'Mz':   action[5]
    })

times = [h['time'] for h in history]
rpms  = [h['rpm']  for h in history]

fig, axes = plt.subplots(2, 1, figsize=(10, 6))
fig.suptitle('NS-29 Capsule Spin-Up Simulation (11 RPM Lunar-g Target)', fontweight='bold')

axes[0].plot(times, rpms, color='steelblue')
axes[0].axhline(TARGET_RPM, color='red', linestyle='--', label=f'Target: {TARGET_RPM} RPM')
axes[0].axhline(TARGET_RPM * 1.02, color='orange', linestyle=':', alpha=0.5, label='+2% bound')
axes[0].axhline(TARGET_RPM * 0.98, color='orange', linestyle=':', alpha=0.5, label='-2% bound')
axes[0].set_ylabel('Capsule Spin Rate (RPM)')
axes[0].legend()
axes[0].set_title('Spin rate vs NS-29 target (11 RPM = lunar-g at locker midpoint)')

axes[1].plot(times, [h['Mz'] for h in history], color='darkorange')
axes[1].axhline(0, color='white', linewidth=0.5)
axes[1].set_ylabel('RCS Torque Mz (N*m)')
axes[1].set_xlabel('Time (s)')

plt.tight_layout()
plt.savefig('ns29_spinup.png', dpi=150)
plt.show()

final_rpm = rpms[-1]
settle_t  = next((h['time'] for h in history if abs(h['rpm'] - TARGET_RPM) < 0.1), None)
print(f"Final spin rate:    {final_rpm:.3f} RPM (target: {TARGET_RPM})")
print(f"Settled to ±0.1 RPM at: t={settle_t:.1f}s" if settle_t else "Did not settle")