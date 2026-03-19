![giphy](https://github.com/user-attachments/assets/f08a45c7-636b-4014-a2d8-cf998619b325)
# Satellite-6DOF-GNC

Full 6DOF rigid body satellite attitude and position control using a Decision Transformer trained on offline trajectory data. Extended from a single-axis pitch PID baseline originally built in MATLAB/Simulink.

## What this is

The original project controlled one axis (pitch) using a PID controller generated from Simulink. This extends that to full 3D position and rotation control, then replaces the PID with a transformer architecture trained to mimic and improve on the baseline controller. The model outputs a probability distribution over actions rather than a single command, which gives a per-step confidence estimate useful for fault detection and planning under uncertainty.

The core architecture is Karpathy's minGPT adapted for continuous state/action spaces. No token vocabulary. Linear projection layers replace the embedding table. The sequence structure is `(return-to-go, state, action)` triplets per timestep, following the Decision Transformer formulation.

## State and action space

| Variable | Dim | Description |
|---|---|---|
| position | 3 | x, y, z (m) |
| velocity | 3 | vx, vy, vz (m/s) |
| quaternion | 4 | q0, q1, q2, q3 (scalar-first) |
| angular rate | 3 | wx, wy, wz (rad/s) |
| **state total** | **13** | |
| force | 3 | Fx, Fy, Fz (N) |
| torque | 3 | Mx, My, Mz (N·m) |
| **action total** | **6** | |

Target orientation is quaternion `[1, 0, 0, 0]` with zero position and angular rate error.

## Architecture

```
Offline trajectories
  500 episodes x 500 steps
  PD controller (Kp_att=5, Kd_att=1) + noise injection
        |
Decision Transformer
  Causal self-attention over (RTG, state, action) triplets
  Continuous input projections: nn.Linear replaces nn.Embedding
  Probabilistic action head: outputs mean + log_std per action dim
  Context window K=20 timesteps
        |
Inference
  Rolling context buffer
  Per-step uncertainty estimate from action std
  Uncertainty flagging for anomaly detection
```

## Inertia parameters (preserved from Simulink baseline)

- Ixx = 8.5 kg·m²
- Iyy = 9.0 kg·m²
- Izz = 10.0 kg·m² (original pitch axis)
- mass = 500 kg
- dt = 0.01s (Euler integration, matches `rt_OneStep` in original `ert_main.c`)

## Quickstart

```bash
pip install torch numpy scipy matplotlib

python data/generate_trajectories.py
python training/train.py
python inference/animate.py
```

`animate.py` opens a live 3D matplotlib window. It loads the trained checkpoint automatically if it exists, and falls back to the PD baseline if not.

## Repo structure

```
dynamics/satellite_6dof.py          full 6DOF rigid body physics
data/generate_trajectories.py       offline data generation via PD controller
model/mingpt_6dof.py                Decision Transformer architecture
training/train.py                   training loop with NLL loss
inference/animate.py                live 3D animation
inference/run_autonomous.py         headless episode runner
evaluation/eval_metrics.py          attitude/position/uncertainty metrics
evaluation/visualize.py             static plot generation
```

## Results

PD baseline achieves ~0.00 deg attitude error with bounded position oscillation. Decision Transformer training is ongoing. Uncertainty output is functional and stable at inference time.

## Background

Built as part of Space Systems Analytics (SSA) research into autonomous GNC systems and onboard intelligence. The longer-term goal is an autonomous GNC/astrodynamics tool integrating learned policies with uncertainty quantification for mission analysis and verification.

## References

- Chen et al., [Decision Transformer: Reinforcement Learning via Sequence Modeling](https://arxiv.org/abs/2106.01385), 2021
- Karpathy, [minGPT](https://github.com/karpathy/minGPT)
- Original 3DOF baseline: [Satellite-Attitude-Control-System-3DOF](https://github.com/Jakayla-R/Satellite-Attitude-Control-System-3DOF)
