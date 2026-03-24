"""
Decision Transformer for 6DOF Satellite GNC
Based on Karpathy's minGPT architecture, adapted for continuous state/action.

Key changes from minGPT:
  - No token vocabulary: replaced nn.Embedding with nn.Linear projections
  - Probabilistic action head (mean + log_std) for uncertainty estimation
  - Input sequence: (return-to-go, state, action) triplets per timestep
"""

import math
import torch
import torch.nn as nn

STATE_DIM  = 13
ACTION_DIM = 6
RTG_DIM    = 1


class CausalSelfAttention(nn.Module):
    """Karpathy's causal self-attention, unchanged from minGPT."""

    def __init__(self, d_model, n_heads, context_len, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_model = d_model
        self.c_attn  = nn.Linear(d_model, 3 * d_model)
        self.c_proj  = nn.Linear(d_model, d_model)
        self.drop    = nn.Dropout(dropout)
        mask = torch.tril(torch.ones(context_len, context_len))
        self.register_buffer("mask", mask)

    def forward(self, x):
        B, T, C = x.shape
        head_dim = C // self.n_heads

        q, k, v = self.c_attn(x).split(self.d_model, dim=2)

        def reshape(t):
            return t.view(B, T, self.n_heads, head_dim).transpose(1, 2)

        q, k, v = reshape(q), reshape(k), reshape(v)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(head_dim)
        att = att.masked_fill(self.mask[:T, :T] == 0, float('-inf'))
        att = torch.softmax(att, dim=-1)
        att = self.drop(att)
        y   = (att @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)


class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, context_len, dropout=0.1):
        super().__init__()
        self.ln1  = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, context_len, dropout)
        self.ln2  = nn.LayerNorm(d_model)
        self.mlp  = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class DecisionTransformer6DOF(nn.Module):
    """
    Decision Transformer for 6DOF satellite attitude + position control.

    Sequence structure per timestep:
        [rtg_t, state_t, action_t]  =>  3 tokens per timestep
        context_len = 3 * K

    Action prediction comes from state tokens only.
    Probabilistic output (mean, std) enables planning under uncertainty.
    """

    def __init__(self, K=20, d_model=128, n_heads=4, n_layers=3, dropout=0.1):
        super().__init__()
        self.K = K
        context_len = 3 * K

        # Continuous input projections (replaces minGPT's nn.Embedding)
        self.embed_rtg    = nn.Linear(RTG_DIM,    d_model)
        self.embed_state  = nn.Linear(STATE_DIM,  d_model)
        self.embed_action = nn.Linear(ACTION_DIM, d_model)

        self.pos_emb = nn.Embedding(context_len, d_model)
        self.drop    = nn.Dropout(dropout)

        self.blocks = nn.Sequential(*[
            TransformerBlock(d_model, n_heads, context_len, dropout)
            for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)

        # Probabilistic action head
        self.action_mean    = nn.Linear(d_model, ACTION_DIM)
        self.action_log_std = nn.Linear(d_model, ACTION_DIM)

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, states, actions, rtgs, deterministic=False):
        """
        Args:
            states:  (B, K, 13)
            actions: (B, K, 6)
            rtgs:    (B, K, 1)
            deterministic: if True, return mean only (no sampling)

        Returns:
            If deterministic: (mean, std)  both (B, K, 6)
            If stochastic:    (action_sample, mean, std)
        """
        B, K, _ = states.shape

        s_emb = self.embed_state(states)    # (B, K, d)
        a_emb = self.embed_action(actions)  # (B, K, d)
        r_emb = self.embed_rtg(rtgs)        # (B, K, d)

        # Interleave tokens: [r0, s0, a0, r1, s1, a1, ...]
        h = torch.stack([r_emb, s_emb, a_emb], dim=2)  # (B, K, 3, d)
        h = h.reshape(B, 3 * K, -1)                     # (B, 3K, d)

        pos = torch.arange(3 * K, device=states.device)
        h   = self.drop(h + self.pos_emb(pos))
        h   = self.blocks(h)
        h   = self.ln_f(h)

        # State tokens are at positions 1, 4, 7, ... (index 3t+1)
        state_tokens = h[:, 1::3, :]  # (B, K, d)

        mean    = self.action_mean(state_tokens)
        log_std = self.action_log_std(state_tokens).clamp(-4, 2)
        std     = log_std.exp()

        if deterministic:
            return mean, std

        # Reparameterization trick for stochastic sampling
        eps    = torch.randn_like(mean)
        action = mean + eps * std
        return action, mean, std

    def get_action(self, states, actions, rtgs):
        """
        Convenience method for inference: returns last timestep action only.
        Args: numpy arrays of shape (K, D)
        Returns: action (6,), uncertainty (6,)
        """
        device = next(self.parameters()).device
        s = torch.FloatTensor(states).unsqueeze(0).to(device)
        a = torch.FloatTensor(actions).unsqueeze(0).to(device)
        r = torch.FloatTensor(rtgs).unsqueeze(0).to(device)

        with torch.no_grad():
            mean, std = self.forward(s, a, r, deterministic=True)

        return mean[0, -1].cpu().numpy(), std[0, -1].cpu().numpy()
