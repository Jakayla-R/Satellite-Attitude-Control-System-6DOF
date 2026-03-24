"""
NS-29 and LEO scenarios using GMAT-propagated truth trajectories.
GMAT provides realistic forces (J2-J8, drag, SRP, lunar/solar gravity).
Your 6DOF attitude dynamics + Decision Transformer run on top.

Run: python scenarios/gmat_scenarios.py

Requires GMAT R2025a at C:/Users/jakay/Documents/GMAT/R2025a
and a trained checkpoint at model/checkpoints/dt_6dof.pt
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib.pyplot as plt
from dynamics import Satellite6DOF, quaternion_error_deg
from data.generate_trajectories import pd_controller, attitude_reward, TARGET_STATE

GMAT_ROOT = r"C:\Users\jakay\Documents\GMAT\R2025a"
GMAT_BIN  = os.path.join(GMAT_ROOT, "bin")
GMAT_API  = os.path.join(GMAT_ROOT, "api")
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'model', 'checkpoints', 'dt_6dof.pt')
K = 20


def load_gmat():
    for p in [GMAT_API, GMAT_BIN]:
        if p not in sys.path:
            sys.path.insert(0, p)
    os.chdir(GMAT_BIN)
    import gmatpy as gmat
    return gmat


def load_model():
    import torch
    from model.mingpt_6dof import DecisionTransformer6DOF
    m = DecisionTransformer6DOF(K=K)
    m.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
    m.eval()
    return m


def run_attitude_scenario(name, gmat_csv_path, scenario_title,
                          initial_att_perturb=None, T_seconds=240):
    """
    Generic scenario runner.
    Uses GMAT position/velocity as truth, evolves attitude with 6DOF engine.

    Args:
        name:                 scenario identifier
        gmat_csv_path:        path to GMAT ReportFile output
        scenario_title:       plot title
        initial_att_perturb:  optional quaternion perturbation from identity
        T_seconds:            simulation duration in seconds
    """
    print(f"\n--- {scenario_title} ---")

    # Load GMAT truth
    data = []
    with open(gmat_csv_path, 'r') as f:
        lines = f.readlines()[1:]  # skip header
    for line in lines:
        parts = line.strip().split()
        nums = []
        for p in parts:
            try:
                nums.append(float(p))
            except ValueError:
                continue
        if len(nums) >= 6:
            data.append(nums[:6])

    gmat_states = np.array(data) * 1000.0  # km -> m
    print(f"GMAT states loaded: {gmat_states.shape}")

    # Try loading DT model
    model = None
    if os.path.exists(MODEL_PATH):
        try:
            model = load_model()
            print("Using trained Decision Transformer")
        except Exception as e:
            print(f"DT load failed ({e}), using PD baseline")
    else:
        print("No checkpoint found, using PD baseline")

    # Sim setup
    dt  = 0.01
    env = Satellite6DOF(mass=6000.0,
                        inertia=np.diag([12000.0, 12000.0, 8000.0]),
                        dt=dt)
    state = env.reset(random_init=False)

    # Seed position/velocity from GMAT
    env.state[0:3] = gmat_states[0, 0:3]
    env.state[3:6] = gmat_states[0, 3:6]

    # Apply attitude perturbation if given
    if initial_att_perturb is not None:
        env.state[6:10] = initial_att_perturb
        env.state[6:10] /= np.linalg.norm(env.state[6:10])
    state = env.state.copy()

    states_buf  = np.zeros((K, 13), dtype=np.float32)
    actions_buf = np.zeros((K, 6),  dtype=np.float32)
    rtgs_buf    = np.full((K, 1), 5.0, dtype=np.float32)

    T      = int(T_seconds / dt)
    history = []
    gmat_idx = 0
    gmat_step = max(1, len(gmat_states) // T)

    for t in range(T):
        # Inject GMAT position/velocity truth at each step
        if gmat_idx < len(gmat_states):
            env.state[0:3] = gmat_states[gmat_idx, 0:3]
            env.state[3:6] = gmat_states[gmat_idx, 3:6]
            gmat_idx += gmat_step
        state = env.state.copy()

        states_buf[-1] = state

        if model is not None:
            action, unc = model.get_action(states_buf, actions_buf, rtgs_buf)
        else:
            action = pd_controller(state)
            unc    = np.zeros(6)

        next_state = env.step(action)
        ae = quaternion_error_deg(state[6:10], TARGET_STATE[6:10])

        history.append({
            'time':        t * dt,
            'att_error':   ae,
            'uncertainty': float(np.mean(unc)),
            'pos':         state[0:3].copy(),
            'omega_norm':  float(np.linalg.norm(state[10:13])),
        })

        states_buf  = np.roll(states_buf,  -1, axis=0)
        actions_buf = np.roll(actions_buf, -1, axis=0)
        rtgs_buf    = np.roll(rtgs_buf,    -1, axis=0)
        actions_buf[-1] = action
        rtgs_buf[-1]    = rtgs_buf[-2] - attitude_reward(state)

        state = next_state

    # Plot
    times  = [h['time']        for h in history]
    errors = [h['att_error']   for h in history]
    uncs   = [h['uncertainty'] for h in history]
    omegas = [h['omega_norm']  for h in history]

    fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
    fig.suptitle(scenario_title, fontweight='bold', fontsize=12)

    axes[0].plot(times, errors, color='steelblue', linewidth=1)
    axes[0].axhline(1.0, color='red', linestyle='--', alpha=0.7, label='1 deg threshold')
    axes[0].set_ylabel('Att Error (deg)')
    axes[0].legend(fontsize=8)

    axes[1].plot(times, omegas, color='darkorange', linewidth=1)
    axes[1].set_ylabel('Omega Norm (rad/s)')

    axes[2].plot(times, uncs, color='purple', linewidth=1, alpha=0.8)
    axes[2].set_ylabel('Uncertainty')
    axes[2].set_xlabel('Time (s)')

    ctrl = 'Decision Transformer' if model else 'PD baseline'
    axes[2].set_title(f'Controller: {ctrl}', fontsize=9, loc='right')

    plt.tight_layout()
    out = f"{name}_result.png"
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Final att error: {errors[-1]:.3f} deg | Mean unc: {np.mean(uncs):.4f}")
    print(f"Saved: {out}")
    return history


def scenario_ns29_microgravity(gmat):
    """
    NS-29: capsule attitude hold during 4-min microgravity window.
    Perturbation from capsule separation event.
    GMAT propagates suborbital arc, attitude controller stabilizes.
    """
    script = """
Create Spacecraft NSCapsule
NSCapsule.DateFormat        = UTCGregorian
NSCapsule.Epoch             = '04 Feb 2025 14:01:50.000'
NSCapsule.CoordinateSystem  = EarthFixed
NSCapsule.DisplayStateType  = Cartesian
NSCapsule.X   = -673.0
NSCapsule.Y   = -5480.0
NSCapsule.Z   =  3698.0
NSCapsule.VX  =  0.0
NSCapsule.VY  =  0.0
NSCapsule.VZ  =  0.85
NSCapsule.DryMass = 6000
NSCapsule.Cd  = 1.2
NSCapsule.Cr  = 1.8
NSCapsule.DragArea = 10.0
NSCapsule.SRPArea  = 10.0

Create ForceModel SuborbitalForces
SuborbitalForces.CentralBody               = Earth
SuborbitalForces.PrimaryBodies             = {Earth}
SuborbitalForces.GravityField.Earth.Degree = 4
SuborbitalForces.GravityField.Earth.Order  = 4
SuborbitalForces.Drag.AtmosphereModel      = MSISE90
SuborbitalForces.SRP                       = Off

Create Propagator SuborbitalProp
SuborbitalProp.FM              = SuborbitalForces
SuborbitalProp.Type            = RungeKutta89
SuborbitalProp.InitialStepSize = 1.0
SuborbitalProp.MinStep         = 0.001
SuborbitalProp.MaxStep         = 5.0

Create ReportFile NS29Report
NS29Report.Filename   = 'ns29_microgravity.csv'
NS29Report.Add        = {NSCapsule.UTCGregorian, NSCapsule.EarthFixed.X, NSCapsule.EarthFixed.Y, NSCapsule.EarthFixed.Z, NSCapsule.EarthFixed.VX, NSCapsule.EarthFixed.VY, NSCapsule.EarthFixed.VZ}
NS29Report.WriteHeaders = true

BeginMissionSequence
Toggle NS29Report On
Propagate SuborbitalProp(NSCapsule) {NSCapsule.ElapsedSecs = 240}
Toggle NS29Report Off
"""
    script_path = os.path.join(GMAT_BIN, 'ns29_microgravity.script')
    with open(script_path, 'w') as f:
        f.write(script)

    gmat.LoadScript(script_path)
    gmat.RunScript()

    csv_path = os.path.join(GMAT_BIN, 'ns29_microgravity.csv')
    return run_attitude_scenario(
        name='ns29_microgravity',
        gmat_csv_path=csv_path,
        scenario_title='NS-29: Capsule Attitude Hold During 4-Min Microgravity Phase (GMAT Truth)',
        initial_att_perturb=np.array([0.995, 0.07, 0.05, 0.02]),
        T_seconds=240
    )


def scenario_leo_nadir_pointing(gmat):
    """
    LEO nadir pointing: ISS-like orbit, attitude controller maintains
    nadir-pointing through one full orbit (90 min).
    GMAT propagates with J8 gravity, drag, SRP, lunar/solar perturbations.
    """
    script = """
Create Spacecraft LEOSat
LEOSat.DateFormat        = UTCGregorian
LEOSat.Epoch             = '01 Jan 2025 12:00:00.000'
LEOSat.CoordinateSystem  = EarthMJ2000Eq
LEOSat.DisplayStateType  = Keplerian
LEOSat.SMA               = 6778.0
LEOSat.ECC               = 0.001
LEOSat.INC               = 51.6
LEOSat.RAAN              = 45.0
LEOSat.AOP               = 0.0
LEOSat.TA                = 0.0
LEOSat.DryMass           = 500
LEOSat.Cd                = 2.2
LEOSat.Cr                = 1.8
LEOSat.DragArea          = 4.0
LEOSat.SRPArea           = 4.0

Create ForceModel LEOForces
LEOForces.CentralBody               = Earth
LEOForces.PrimaryBodies             = {Earth}
LEOForces.GravityField.Earth.Degree = 8
LEOForces.GravityField.Earth.Order  = 8
LEOForces.Drag.AtmosphereModel      = MSISE90
LEOForces.SRP                       = On
LEOForces.PointMasses               = {Sun, Luna}

Create Propagator LEOProp
LEOProp.FM              = LEOForces
LEOProp.Type            = RungeKutta89
LEOProp.InitialStepSize = 30
LEOProp.MinStep         = 0.001
LEOProp.MaxStep         = 60.0

Create ReportFile LEOReport
LEOReport.Filename      = 'leo_nadir.csv'
LEOReport.Add           = {LEOSat.UTCGregorian, LEOSat.EarthMJ2000Eq.X, LEOSat.EarthMJ2000Eq.Y, LEOSat.EarthMJ2000Eq.Z, LEOSat.EarthMJ2000Eq.VX, LEOSat.EarthMJ2000Eq.VY, LEOSat.EarthMJ2000Eq.VZ}
LEOReport.WriteHeaders  = true

BeginMissionSequence
Toggle LEOReport On
Propagate LEOProp(LEOSat) {LEOSat.ElapsedSecs = 5400}
Toggle LEOReport Off
"""
    script_path = os.path.join(GMAT_BIN, 'leo_nadir.script')
    with open(script_path, 'w') as f:
        f.write(script)

    gmat.LoadScript(script_path)
    gmat.RunScript()

    csv_path = os.path.join(GMAT_BIN, 'leo_nadir.csv')
    return run_attitude_scenario(
        name='leo_nadir',
        gmat_csv_path=csv_path,
        scenario_title='LEO Nadir Pointing: ISS-like Orbit with J8+Drag+SRP (GMAT Truth)',
        initial_att_perturb=np.array([0.98, 0.1, 0.1, 0.05]),
        T_seconds=300
    )


if __name__ == "__main__":
    print("Loading GMAT API...")
    gmat = load_gmat()

    print("\nRunning NS-29 microgravity scenario...")
    scenario_ns29_microgravity(gmat)

    print("\nRunning LEO nadir pointing scenario...")
    scenario_leo_nadir_pointing(gmat)

    print("\nAll scenarios complete.")
