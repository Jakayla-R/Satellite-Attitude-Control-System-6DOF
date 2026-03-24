"""
Run autonomous 6DOF attitude control using trained Decision Transformer.

Run: python inference/run_autonomous.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import torch

from dynamics import Satellite6DOF
from model.mingpt_6dof import DecisionTransformer6DOF
from data.generate_trajectories import attitude_reward, TARGET_STATE
from evaluation.eval_metrics import evaluate_episode


def run_autonomous(
    model_path   = None,
    K            = 20,
    T            = 500,
    target_rtg   = 5.0,
    uncertainty_threshold = 3.0,
    seed         = 0,
):
    """
    Run one autonomous episode.

    Args:
        model_path:            path to saved .pt checkpoint
        K:                     context window length (timesteps)
        T:                     episode length
        target_rtg:            desired return-to-go (higher = more ambitious)
        uncertainty_threshold: if mean action std exceeds this, flag the step
        seed:                  random seed for initial conditions

    Returns:
        history: list of dicts with state, action, uncertainty per step
    """
    base = os.path.dirname(__file__)
    if model_path is None:
        model_path = os.path.join(base, '..', 'model', 'checkpoints', 'dt_6dof.pt')

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"No checkpoint found at {model_path}. Run training/train.py first."
        )

    device = torch.device("cpu")
    model  = DecisionTransformer6DOF(K=K)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f"Loaded model from {model_path}")

    env   = Satellite6DOF()
    state = env.reset(random_init=True, seed=seed)

    # Rolling context buffers (start with zeros, fill in as episode runs)
    states  = np.zeros((K, 13), dtype=np.float32)
    actions = np.zeros((K, 6),  dtype=np.float32)
    rtgs    = np.full((K, 1), target_rtg, dtype=np.float32)

    history     = []
    high_uncertainty_steps = 0

    for t in range(T):
        states[-1] = state

        action, uncertainty = model.get_action(states, actions, rtgs)

        # Planning under uncertainty: flag high-uncertainty steps
        mean_unc = uncertainty.mean()
        if mean_unc > uncertainty_threshold:
            high_uncertainty_steps += 1

        next_state = env.step(action)

        history.append({
            "state":       state.copy(),
            "action":      action.copy(),
            "uncertainty": uncertainty.copy(),
            "flagged":     mean_unc > uncertainty_threshold,
        })

        # Shift context buffers forward
        states  = np.roll(states,  -1, axis=0)
        actions = np.roll(actions, -1, axis=0)
        rtgs    = np.roll(rtgs,    -1, axis=0)
        actions[-1] = action
        rtgs[-1]    = rtgs[-2] - attitude_reward(state)

        state = next_state

    print(f"\nEpisode complete ({T} steps)")
    print(f"High-uncertainty steps: {high_uncertainty_steps}/{T} "
          f"({100*high_uncertainty_steps/T:.1f}%)")

    return history


if __name__ == "__main__":
    history = run_autonomous()

    metrics = evaluate_episode(history, TARGET_STATE)
    print(f"\n--- Evaluation ---")
    print(f"Mean attitude error : {metrics['att_error_mean_deg']:.3f} deg")
    print(f"Final attitude error: {metrics['att_error_final_deg']:.3f} deg")
    print(f"Mean position error : {metrics['pos_error_mean_m']:.3f} m")
    print(f"Final position error: {metrics['pos_error_final_m']:.3f} m")
    print(f"Mean uncertainty    : {metrics['mean_uncertainty']:.4f}")

    from evaluation.visualize import plot_attitude_error, plot_trajectory_3d, plot_quaternion_components
    plot_attitude_error(metrics, save_path="attitude_error.png")
    plot_trajectory_3d(history,  save_path="trajectory_3d.png")
    plot_quaternion_components(history, save_path="quaternion_history.png")
    print("\nPlots saved.")
