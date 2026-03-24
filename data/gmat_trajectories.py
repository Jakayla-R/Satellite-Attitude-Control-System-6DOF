"""
GMAT-powered trajectory generation for 6DOF Decision Transformer training.
Replaces the simple PD controller rollouts with physically realistic
orbital/suborbital propagation using GMAT R2022a.

Run: python data/gmat_trajectories.py

Requires GMAT R2022a installed at C:/Users/jakay/Documents/GMAT/R2025a
"""

import sys
import os
import numpy as np

# ---- GMAT API path setup ----
GMAT_ROOT = r"C:\Users\jakay\Documents\GMAT\R2025a"
GMAT_API  = os.path.join(GMAT_ROOT, "api")
GMAT_BIN  = os.path.join(GMAT_ROOT, "bin")

for p in [GMAT_API, GMAT_BIN]:
    if p not in sys.path:
        sys.path.insert(0, p)

os.chdir(GMAT_BIN)  # GMAT needs to run from bin directory

try:
    import gmatpy as gmat
    print("GMAT API loaded successfully.")
except ImportError as e:
    raise ImportError(
        f"Could not import gmatpy: {e}\n"
        f"Check that {GMAT_API} exists and contains gmatpy files."
    )

# ---- Scenario definitions ----

NS29_SCRIPT = """
%----------------------------------------
% NS-29 Suborbital Trajectory
% Based on public Blue Origin flight data:
%   - Launch from West Texas (31.4N, 104.8W)
%   - Peak altitude ~107 km
%   - Total flight ~10 min
%   - Capsule separates at ~110s / ~38km
%----------------------------------------

Create Spacecraft NSCapsule
NSCapsule.DateFormat        = UTCGregorian
NSCapsule.Epoch             = '04 Feb 2025 14:00:00.000'
NSCapsule.CoordinateSystem  = EarthFixed
NSCapsule.DisplayStateType  = Cartesian

% West Texas launch site converted to ECEF (approx)
NSCapsule.X   = -673.0
NSCapsule.Y   = -5480.0
NSCapsule.Z   =  3298.0
NSCapsule.VX  =  0.0
NSCapsule.VY  =  0.0
NSCapsule.VZ  =  1.05

NSCapsule.DryMass          = 6000
NSCapsule.Cd               = 1.2
NSCapsule.Cr               = 1.8
NSCapsule.DragArea         = 10.0
NSCapsule.SRPArea          = 10.0

Create ForceModel SuborbitalForces
SuborbitalForces.CentralBody               = Earth
SuborbitalForces.PrimaryBodies             = {Earth}
SuborbitalForces.GravityField.Earth.Degree = 4
SuborbitalForces.GravityField.Earth.Order  = 4
SuborbitalForces.Drag.AtmosphereModel      = MSISE90
SuborbitalForces.PointMasses               = {}
SuborbitalForces.SRP                       = Off

Create Propagator SuborbitalProp
SuborbitalProp.FM           = SuborbitalForces
SuborbitalProp.Type         = RungeKutta89
SuborbitalProp.InitialStepSize = 0.5
SuborbitalProp.MinStep      = 0.001
SuborbitalProp.MaxStep      = 2.0
SuborbitalProp.MaxStepAttempts = 50

Create ReportFile TrajectoryReport
TrajectoryReport.Filename   = 'ns29_trajectory.csv'
TrajectoryReport.Add        = {NSCapsule.UTCGregorian, NSCapsule.EarthFixed.X, NSCapsule.EarthFixed.Y, NSCapsule.EarthFixed.Z, NSCapsule.EarthFixed.VX, NSCapsule.EarthFixed.VY, NSCapsule.EarthFixed.VZ, NSCapsule.Altitude}
TrajectoryReport.WriteHeaders = true

BeginMissionSequence

Toggle TrajectoryReport On
Propagate SuborbitalProp(NSCapsule) {NSCapsule.ElapsedSecs = 600}
Toggle TrajectoryReport Off
"""

LEO_SCRIPT = """
%----------------------------------------
% Low Earth Orbit scenario
% ISS-like orbit for attitude control validation
% 400km circular, 51.6 deg inclination
%----------------------------------------

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
LEOReport.Filename      = 'leo_trajectory.csv'
LEOReport.Add           = {LEOSat.UTCGregorian, LEOSat.EarthMJ2000Eq.X, LEOSat.EarthMJ2000Eq.Y, LEOSat.EarthMJ2000Eq.Z, LEOSat.EarthMJ2000Eq.VX, LEOSat.EarthMJ2000Eq.VY, LEOSat.EarthMJ2000Eq.VZ, LEOSat.Altitude, LEOSat.Latitude, LEOSat.Longitude}
LEOReport.WriteHeaders  = true

BeginMissionSequence

Toggle LEOReport On
Propagate LEOProp(LEOSat) {LEOSat.ElapsedSecs = 5400}
Toggle LEOReport Off
"""


def run_gmat_scenario(script_text, scenario_name):
    """Run a GMAT script string and return output CSV path."""
    print(f"\nRunning GMAT scenario: {scenario_name}")

    # Write script to temp file
    script_path = os.path.join(GMAT_BIN, f"{scenario_name}.script")
    with open(script_path, 'w') as f:
        f.write(script_text)

    # Load and run via API
    gmat.LoadScript(script_path)
    success = gmat.RunScript()

    if not success:
        raise RuntimeError(f"GMAT scenario {scenario_name} failed to run.")

    # Find output CSV
    csv_path = os.path.join(GMAT_BIN, f"{scenario_name.replace('_script','')}.csv")
    if not os.path.exists(csv_path):
        # GMAT writes to bin dir by default
        csv_path = os.path.join(GMAT_BIN, f"{scenario_name}.csv")

    print(f"GMAT output: {csv_path}")
    return csv_path


def parse_gmat_csv(csv_path, max_rows=None):
    """
    Parse GMAT ReportFile output into numpy array.
    Returns: (times, states) where states is (N, 6) [x,y,z,vx,vy,vz] in km and km/s
    """
    data = []
    with open(csv_path, 'r') as f:
        lines = f.readlines()

    # Skip header
    for line in lines[1:]:
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        try:
            # Skip epoch string columns, grab numerics
            nums = [float(p) for p in parts if _is_float(p)]
            if len(nums) >= 6:
                data.append(nums)
        except ValueError:
            continue
        if max_rows and len(data) >= max_rows:
            break

    arr = np.array(data)
    print(f"Parsed {len(arr)} states from {csv_path}")
    return arr


def _is_float(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def gmat_states_to_training_data(gmat_array, dt=0.01, K=500):
    """
    Convert GMAT output (km, km/s) to training trajectory format.
    Builds (K, 19) windows: [state(13), action(6)]

    State format matches your 6DOF dynamics:
    [x, y, z, vx, vy, vz, q0, q1, q2, q3, wx, wy, wz]

    GMAT gives position/velocity. Attitude and omega are initialized
    to identity/zero and evolved forward using your dynamics engine.
    """
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from dynamics import Satellite6DOF
    from data.generate_trajectories import pd_controller

    # Convert km -> m, km/s -> m/s
    positions = gmat_array[:, :3] * 1000.0
    velocities = gmat_array[:, 3:6] * 1000.0

    n_steps = min(len(positions), K)
    env = Satellite6DOF(dt=dt)
    state = env.reset(random_init=False)

    # Seed with GMAT position/velocity
    env.state[0:3] = positions[0]
    env.state[3:6] = velocities[0]
    state = env.state.copy()

    trajectory = np.zeros((n_steps, 19), dtype=np.float32)

    for t in range(n_steps):
        # Update position/velocity from GMAT truth (hybrid approach)
        # Attitude evolved by your dynamics engine
        env.state[0:3] = positions[min(t, len(positions)-1)]
        env.state[3:6] = velocities[min(t, len(velocities)-1)]
        state = env.state.copy()

        action = pd_controller(state)
        trajectory[t, :13] = state
        trajectory[t, 13:]  = action
        env.step(action)

    return trajectory


def generate_gmat_training_data(output_path=None):
    """
    Full pipeline: run GMAT, parse output, convert to training format, save.
    """
    base = os.path.dirname(__file__)
    if output_path is None:
        output_path = os.path.join(base, 'gmat_trajectories.npy')

    all_trajs = []

    # Scenario 1: NS-29 suborbital
    try:
        csv = run_gmat_scenario(NS29_SCRIPT, 'ns29_trajectory')
        arr = parse_gmat_csv(csv)
        if len(arr) > 50:
            traj = gmat_states_to_training_data(arr, K=500)
            all_trajs.append(traj)
            print(f"NS-29 trajectory: {traj.shape}")
    except Exception as e:
        print(f"NS-29 scenario failed: {e}")

    # Scenario 2: LEO orbit
    try:
        csv = run_gmat_scenario(LEO_SCRIPT, 'leo_trajectory')
        arr = parse_gmat_csv(csv)
        if len(arr) > 50:
            traj = gmat_states_to_training_data(arr, K=500)
            all_trajs.append(traj)
            print(f"LEO trajectory: {traj.shape}")
    except Exception as e:
        print(f"LEO scenario failed: {e}")

    if not all_trajs:
        raise RuntimeError("No GMAT trajectories generated successfully.")

    data = np.array(all_trajs)  # (n_scenarios, K, 19)
    np.save(output_path, data)
    print(f"\nSaved {data.shape} GMAT trajectories -> {output_path}")
    return data


if __name__ == "__main__":
    generate_gmat_training_data()
