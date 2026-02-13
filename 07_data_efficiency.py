"""
Step 7: Data Efficiency Experiment

How much demonstration data do we need?
Train the SAME model with different amounts of data and compare.

This is critical practical knowledge:
- Real robot data is expensive to collect
- Understanding the data-performance curve saves time and money
"""

import gymnasium as gym
import gymnasium_robotics
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

gym.register_envs(gymnasium_robotics)

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Device: {device}")


class BCPolicy(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, act_dim),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.net(x)


# ============================================================
# Collect a LARGE pool of expert data (FetchReach)
# ============================================================
print("Collecting expert data pool...")


def expert_policy(obs):
    direction = obs["desired_goal"] - obs["achieved_goal"]
    action = np.zeros(4)
    action[:3] = np.clip(direction * 10, -1, 1)
    return action


env = gym.make("FetchReach-v4", render_mode=None)
all_obs, all_act = [], []

for ep in range(500):
    obs, _ = env.reset()
    for step in range(50):
        obs_vec = np.concatenate([obs["observation"], obs["desired_goal"]])
        action = expert_policy(obs)
        all_obs.append(obs_vec)
        all_act.append(action)
        obs, _, term, trunc, _ = env.step(action)
        if term or trunc:
            break

env.close()

all_obs = np.array(all_obs)
all_act = np.array(all_act)
print(f"Total data pool: {len(all_obs)} frames")

# ============================================================
# Train with different data amounts
# ============================================================
data_fractions = [0.01, 0.05, 0.1, 0.25, 0.5, 1.0]
results = {}

print("\n" + "=" * 60)
print(f"{'Data %':>8} | {'Frames':>8} | {'Loss':>10} | {'Success Rate':>14}")
print("=" * 60)

for frac in data_fractions:
    n = max(int(len(all_obs) * frac), 50)
    obs_sub = torch.FloatTensor(all_obs[:n]).to(device)
    act_sub = torch.FloatTensor(all_act[:n]).to(device)

    policy = BCPolicy(all_obs.shape[1], all_act.shape[1]).to(device)
    optimizer = optim.Adam(policy.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    # Train
    for epoch in range(100):
        idx = torch.randperm(n)
        for i in range(0, n, 256):
            batch = idx[i : i + 256]
            pred = policy(obs_sub[batch])
            loss = loss_fn(pred, act_sub[batch])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # Evaluate
    policy.eval()
    env = gym.make("FetchReach-v4", render_mode=None)
    successes = 0
    eval_eps = 50

    for ep in range(eval_eps):
        obs, _ = env.reset()
        for step in range(50):
            obs_vec = np.concatenate([obs["observation"], obs["desired_goal"]])
            t = torch.FloatTensor(obs_vec).unsqueeze(0).to(device)
            with torch.no_grad():
                action = policy(t).cpu().numpy()[0]
            obs, _, term, trunc, info = env.step(action)
            if info.get("is_success", False):
                successes += 1
                break
            if term or trunc:
                break

    env.close()
    sr = successes / eval_eps * 100
    results[frac] = sr

    final_loss = loss.item()
    print(f"{frac * 100:>7.0f}% | {n:>8d} | {final_loss:>10.6f} | {sr:>12.0f}%")

# ============================================================
# Summary
# ============================================================
print("\n" + "=" * 60)
print("Key Takeaways")
print("=" * 60)

min_good = None
for frac in data_fractions:
    if results[frac] >= 80:
        min_good = frac
        break

if min_good:
    print(f"  -> {min_good * 100:.0f}% of data ({int(len(all_obs) * min_good)} frames) is enough for 80%+ success")
else:
    print(f"  -> Even 100% data wasn't enough for 80% success on this config")

print(f"  -> More data generally = better performance")
print(f"  -> But there are diminishing returns after a certain point")
print(f"  -> For real robots: find the sweet spot to minimize data collection cost!")
