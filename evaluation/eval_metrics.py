"""
Evaluation metrics for 6DOF attitude control episodes.
"""

import numpy as np
from dynamics import quaternion_error_deg


def evaluate_episode(history, target):
    """
    Compute performance metrics over a full episode.

    Args:
        history: list of step dicts from run_autonomous()
        target:  target state array (13,)

    Returns:
        dict of scalar and array metrics
    """
    q_ref = target[6:10]

    att_errors    = []
    pos_errors    = []
    vel_errors    = []
    omega_norms   = []
    uncertainties = []
    flagged_steps = []

    for step in history:
        s = step["state"]
        att_errors.append(quaternion_error_deg(s[6:10], q_ref))
        pos_errors.append(np.linalg.norm(s[0:3] - target[0:3]))
        vel_errors.append(np.linalg.norm(s[3:6] - target[3:6]))
        omega_norms.append(np.linalg.norm(s[10:13]))
        uncertainties.append(step["uncertainty"].mean())
        flagged_steps.append(float(step.get("flagged", False)))

    return {
        "att_error_mean_deg":   np.mean(att_errors),
        "att_error_final_deg":  att_errors[-1],
        "pos_error_mean_m":     np.mean(pos_errors),
        "pos_error_final_m":    pos_errors[-1],
        "vel_error_mean":       np.mean(vel_errors),
        "omega_norm_mean":      np.mean(omega_norms),
        "mean_uncertainty":     np.mean(uncertainties),
        "pct_flagged":          100 * np.mean(flagged_steps),
        # Arrays for plotting
        "att_errors":           att_errors,
        "pos_errors":           pos_errors,
        "vel_errors":           vel_errors,
        "omega_norms":          omega_norms,
        "uncertainties":        uncertainties,
        "flagged_steps":        flagged_steps,
    }
