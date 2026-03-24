"""
Train the Decision Transformer on offline 6DOF trajectory data.

Run: python training/train.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import torch
import torch.nn as nn

from model.mingpt_6dof import DecisionTransformer6DOF
from data.generate_trajectories import attitude_reward, TARGET_STATE


def compute_rtg(states, gamma=0.99):
    """Compute return-to-go sequence from state trajectory."""
    rewards = np.array([attitude_reward(s) for s in states])
    rtg = np.zeros_like(rewards)
    running = 0.0
    for t in reversed(range(len(rewards))):
        running  = rewards[t] + gamma * running
        rtg[t]   = running
    return rtg


def train(
    data_path   = None,
    epochs      = 1000,
    batch_size  = 64,
    K           = 20,
    lr          = 1e-4,
    save_path   = None,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    # Resolve paths relative to this file
    base = os.path.dirname(__file__)
    if data_path is None:
        data_path = os.path.join(base, '..', 'data', 'trajectories.npy')
    if save_path is None:
        save_path = os.path.join(base, '..', 'model', 'checkpoints', 'dt_6dof.pt')

    if not os.path.exists(data_path):
        print("Trajectory data not found. Generating now...")
        from data.generate_trajectories import generate_trajectories
        data = generate_trajectories(n_traj=500, T=200)
        np.save(data_path, data)
    else:
        print(f"Loading data from {data_path}")
        data = np.load(data_path)

    print(f"Data shape: {data.shape}")  # (n_traj, T, 19)
    n_traj, T, _ = data.shape

    model = DecisionTransformer6DOF(K=K).to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    best_loss = float('inf')

    for epoch in range(epochs):
        model.train()

        idx_traj = np.random.randint(0, n_traj,   batch_size)
        idx_time = np.random.randint(0, T - K - 1, batch_size)

        states_batch  = []
        actions_batch = []
        rtgs_batch    = []

        for i, t in zip(idx_traj, idx_time):
            window  = data[i, t:t+K]
            states  = window[:, :13]
            actions = window[:, 13:]
            rtg     = compute_rtg(states)

            states_batch.append(states)
            actions_batch.append(actions)
            rtgs_batch.append(rtg[:, None])

        s = torch.FloatTensor(np.array(states_batch)).to(device)
        a = torch.FloatTensor(np.array(actions_batch)).to(device)
        r = torch.FloatTensor(np.array(rtgs_batch)).to(device)

        action_sample, mean, std = model(s, a, r)

        # Gaussian NLL loss: trains probabilistic distribution over actions
        dist = torch.distributions.Normal(mean, std + 1e-6)
        loss = -dist.log_prob(a).mean()

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1:4d}/{epochs} | Loss: {loss.item():.4f} | LR: {sched.get_last_lr()[0]:.2e}")

        if loss.item() < best_loss:
            best_loss = loss.item()
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(model.state_dict(), save_path)

    print(f"\nTraining complete. Best loss: {best_loss:.4f}")
    print(f"Model saved to: {save_path}")


if __name__ == "__main__":
    train()
