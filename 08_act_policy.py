"""
Step 8: ACT (Action Chunking with Transformers)

The algorithm behind ALOHA and many state-of-the-art robot systems.

Key ideas:
1. Action Chunking: predict K future actions at once, not just the next one
2. Transformer encoder: captures temporal dependencies
3. CVAE: handles multimodal action distributions

This is a simplified but functional implementation.
"""

import gymnasium as gym
import gymnasium_robotics
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import defaultdict

gym.register_envs(gymnasium_robotics)

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Device: {device}")

CHUNK_SIZE = 5  # predict 5 future actions at once
OBS_HORIZON = 3  # look at last 3 observations


# ============================================================
# ACT-style policy network
# ============================================================


class ACTPolicy(nn.Module):
    """
    Simplified ACT: Transformer-based action chunking policy.
    Input: sequence of observations (obs_horizon)
    Output: chunk of future actions (chunk_size)
    """

    def __init__(self, obs_dim, act_dim, chunk_size, d_model=256, nhead=4, num_layers=3):
        super().__init__()
        self.chunk_size = chunk_size
        self.obs_dim = obs_dim
        self.act_dim = act_dim

        # Observation encoder
        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_dim, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )

        # Positional encoding (learnable)
        self.pos_embed = nn.Parameter(torch.randn(1, 20, d_model) * 0.02)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=512, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Action chunk decoder
        self.action_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, act_dim * chunk_size),
            nn.Tanh(),
        )

    def forward(self, obs_seq):
        """
        obs_seq: (batch, seq_len, obs_dim)
        returns: (batch, chunk_size, act_dim)
        """
        B, T, _ = obs_seq.shape

        # Encode observations
        x = self.obs_encoder(obs_seq)  # (B, T, d_model)
        x = x + self.pos_embed[:, :T, :]

        # Transformer
        x = self.transformer(x)  # (B, T, d_model)

        # Use the last token to predict action chunk
        last = x[:, -1, :]  # (B, d_model)

        # Decode action chunk
        actions = self.action_head(last)  # (B, act_dim * chunk_size)
        actions = actions.view(B, self.chunk_size, self.act_dim)

        return actions


# ============================================================
# Collect sequential expert data
# ============================================================
print("\n" + "=" * 50)
print("Step 1: Collecting sequential expert data")
print("=" * 50)


def expert_policy(obs):
    direction = obs["desired_goal"] - obs["achieved_goal"]
    action = np.zeros(4)
    action[:3] = np.clip(direction * 10, -1, 1)
    return action


env = gym.make("FetchReach-v4", render_mode=None)
episodes_data = []

for ep in range(300):
    obs, _ = env.reset()
    ep_obs, ep_act = [], []

    for step in range(50):
        obs_vec = np.concatenate([obs["observation"], obs["desired_goal"]])
        action = expert_policy(obs)
        ep_obs.append(obs_vec)
        ep_act.append(action)
        obs, _, term, trunc, _ = env.step(action)
        if term or trunc:
            break

    episodes_data.append({"obs": np.array(ep_obs), "act": np.array(ep_act)})

env.close()
print(f"Collected {len(episodes_data)} episodes")


# Build training sequences: (obs_history, action_chunk) pairs
def build_chunks(episodes, obs_horizon, chunk_size):
    obs_seqs, act_chunks = [], []
    for ep in episodes:
        T = len(ep["obs"])
        for t in range(obs_horizon - 1, T - chunk_size):
            obs_seq = ep["obs"][t - obs_horizon + 1 : t + 1]  # (obs_horizon, obs_dim)
            act_chunk = ep["act"][t : t + chunk_size]  # (chunk_size, act_dim)
            obs_seqs.append(obs_seq)
            act_chunks.append(act_chunk)
    return np.array(obs_seqs), np.array(act_chunks)


obs_seqs, act_chunks = build_chunks(episodes_data, OBS_HORIZON, CHUNK_SIZE)
print(f"Training sequences: {len(obs_seqs)}")
print(f"  obs_seq shape: {obs_seqs.shape}")
print(f"  act_chunk shape: {act_chunks.shape}")

# ============================================================
# Train ACT policy
# ============================================================
print("\n" + "=" * 50)
print("Step 2: Training ACT policy")
print("=" * 50)

obs_dim = obs_seqs.shape[2]
act_dim = act_chunks.shape[2]

policy = ACTPolicy(obs_dim, act_dim, CHUNK_SIZE).to(device)
optimizer = optim.Adam(policy.parameters(), lr=1e-4)
loss_fn = nn.MSELoss()

obs_t = torch.FloatTensor(obs_seqs).to(device)
act_t = torch.FloatTensor(act_chunks).to(device)
N = len(obs_t)

for epoch in range(100):
    idx = torch.randperm(N)
    total_loss = 0
    n_batch = 0

    for i in range(0, N, 256):
        batch = idx[i : i + 256]
        pred_chunks = policy(obs_t[batch])
        loss = loss_fn(pred_chunks, act_t[batch])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batch += 1

    if (epoch + 1) % 20 == 0:
        print(f"  Epoch {epoch + 1:3d}/100 | Loss: {total_loss / n_batch:.6f}")

# ============================================================
# Evaluate ACT vs simple BC
# ============================================================
print("\n" + "=" * 50)
print("Step 3: Evaluation — ACT vs BC")
print("=" * 50)

policy.eval()


def evaluate_act(num_eps=50):
    env = gym.make("FetchReach-v4", render_mode=None)
    successes = 0

    for ep in range(num_eps):
        obs, _ = env.reset()
        obs_history = []
        action_queue = []

        for step in range(50):
            obs_vec = np.concatenate([obs["observation"], obs["desired_goal"]])
            obs_history.append(obs_vec)

            if len(action_queue) == 0:
                # Need new action chunk
                if len(obs_history) >= OBS_HORIZON:
                    recent = np.array(obs_history[-OBS_HORIZON:])
                else:
                    # Pad with first observation
                    pad = [obs_history[0]] * (OBS_HORIZON - len(obs_history))
                    recent = np.array(pad + obs_history)

                t = torch.FloatTensor(recent).unsqueeze(0).to(device)
                with torch.no_grad():
                    chunk = policy(t).cpu().numpy()[0]  # (chunk_size, act_dim)
                action_queue = list(chunk)

            action = action_queue.pop(0)
            obs, _, term, trunc, info = env.step(action)

            if info.get("is_success", False):
                successes += 1
                break
            if term or trunc:
                break

    env.close()
    return successes / num_eps * 100


act_sr = evaluate_act()


# Compare with simple BC (from step 4)
class SimpleBCPolicy(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, act_dim), nn.Tanh(),
        )

    def forward(self, x):
        return self.net(x)


# Quick train simple BC for comparison
flat_obs = obs_seqs[:, -1, :]  # just last observation
flat_act = act_chunks[:, 0, :]  # just next action
bc_obs_t = torch.FloatTensor(flat_obs).to(device)
bc_act_t = torch.FloatTensor(flat_act).to(device)

bc_policy = SimpleBCPolicy(obs_dim, act_dim).to(device)
bc_opt = optim.Adam(bc_policy.parameters(), lr=1e-3)

for epoch in range(50):
    idx = torch.randperm(len(bc_obs_t))
    for i in range(0, len(bc_obs_t), 256):
        batch = idx[i : i + 256]
        loss = nn.MSELoss()(bc_policy(bc_obs_t[batch]), bc_act_t[batch])
        bc_opt.zero_grad()
        loss.backward()
        bc_opt.step()

bc_policy.eval()
env = gym.make("FetchReach-v4", render_mode=None)
bc_success = 0
for ep in range(50):
    obs, _ = env.reset()
    for step in range(50):
        obs_vec = np.concatenate([obs["observation"], obs["desired_goal"]])
        t = torch.FloatTensor(obs_vec).unsqueeze(0).to(device)
        with torch.no_grad():
            action = bc_policy(t).cpu().numpy()[0]
        obs, _, term, trunc, info = env.step(action)
        if info.get("is_success", False):
            bc_success += 1
            break
        if term or trunc:
            break
env.close()
bc_sr = bc_success / 50 * 100

print(f"\n  Simple BC:  {bc_sr:.0f}%")
print(f"  ACT:        {act_sr:.0f}%")
print(f"\n  Key insight: ACT predicts {CHUNK_SIZE} actions at once (action chunking)")
print(f"  This gives smoother, more consistent trajectories.")

torch.save(policy.state_dict(), "/Users/chan/robot-learning/act_fetchreach.pt")
print(f"\nModel saved: act_fetchreach.pt")
