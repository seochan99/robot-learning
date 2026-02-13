"""
Step 6: Behavioral Cloning on FetchPush — Hitting the Limits

FetchPush is harder than FetchReach:
- Robot must PUSH an object to a target position
- Requires contact-rich manipulation
- Simple BC starts to struggle here

This demonstrates WHY we need better algorithms (ACT, Diffusion Policy).
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

# ============================================================
# Expert policy for FetchPush (heuristic-based)
# ============================================================


def expert_push_policy(obs):
    """
    Two-phase expert:
    1. Move gripper above the object
    2. Push the object toward the target
    """
    observation = obs["observation"]
    grip_pos = observation[:3]
    object_pos = observation[3:6]
    target_pos = obs["desired_goal"]

    action = np.zeros(4)
    object_to_target = target_pos - object_pos
    grip_to_object = object_pos - grip_pos

    # Phase 1: approach the object from the push direction
    push_direction = object_to_target[:2] / (np.linalg.norm(object_to_target[:2]) + 1e-6)
    approach_pos = object_pos.copy()
    approach_pos[:2] -= push_direction * 0.05  # behind the object
    approach_pos[2] = object_pos[2]  # same height

    dist_to_object = np.linalg.norm(grip_pos[:2] - object_pos[:2])

    if dist_to_object > 0.05 or abs(grip_pos[2] - object_pos[2]) > 0.02:
        # Move to approach position
        direction = approach_pos - grip_pos
        action[:3] = np.clip(direction * 10, -1, 1)
    else:
        # Phase 2: push toward target
        action[:3] = np.clip(object_to_target * 5, -1, 1)
        action[2] = np.clip((object_pos[2] - grip_pos[2]) * 10, -1, 1)

    action[3] = 1.0  # keep gripper open
    return action


# ============================================================
# Collect expert demonstrations
# ============================================================
print("\n" + "=" * 50)
print("Step 1: Collecting expert demos on FetchPush")
print("=" * 50)

env = gym.make("FetchPush-v4", render_mode=None)
expert_data = defaultdict(list)
num_episodes = 500
successes = 0

for ep in range(num_episodes):
    obs, info = env.reset()
    episode_obs = []
    episode_actions = []
    ep_success = False

    for step in range(50):
        obs_vector = np.concatenate([
            obs["observation"], obs["achieved_goal"], obs["desired_goal"]
        ])
        action = expert_push_policy(obs)

        episode_obs.append(obs_vector)
        episode_actions.append(action)

        obs, reward, terminated, truncated, info = env.step(action)

        if info.get("is_success", False):
            ep_success = True

        if terminated or truncated:
            break

    if ep_success:
        successes += 1
        expert_data["observations"].extend(episode_obs)
        expert_data["actions"].extend(episode_actions)

env.close()

expert_sr = successes / num_episodes * 100
print(f"Expert success rate: {expert_sr:.0f}%")
print(f"Collected {len(expert_data['observations'])} frames from successful episodes")

# ============================================================
# Train BC policy
# ============================================================
print("\n" + "=" * 50)
print("Step 2: Training BC on FetchPush")
print("=" * 50)

obs_array = np.array(expert_data["observations"])
act_array = np.array(expert_data["actions"])


class BCPolicy(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, act_dim),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.net(x)


obs_tensor = torch.FloatTensor(obs_array).to(device)
act_tensor = torch.FloatTensor(act_array).to(device)

policy = BCPolicy(obs_array.shape[1], act_array.shape[1]).to(device)
optimizer = optim.Adam(policy.parameters(), lr=3e-4)
loss_fn = nn.MSELoss()

for epoch in range(100):
    indices = torch.randperm(len(obs_tensor))
    total_loss = 0
    n = 0

    for i in range(0, len(obs_tensor), 512):
        idx = indices[i : i + 512]
        pred = policy(obs_tensor[idx])
        loss = loss_fn(pred, act_tensor[idx])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n += 1

    if (epoch + 1) % 20 == 0:
        print(f"  Epoch {epoch + 1:3d}/100 | Loss: {total_loss / n:.6f}")

# ============================================================
# Evaluate
# ============================================================
print("\n" + "=" * 50)
print("Step 3: Evaluation")
print("=" * 50)

policy.eval()
env = gym.make("FetchPush-v4", render_mode=None)


def evaluate(policy_fn, name, num_eps=50):
    s = 0
    for ep in range(num_eps):
        obs, info = env.reset()
        for step in range(50):
            action = policy_fn(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            if info.get("is_success", False):
                s += 1
                break
            if terminated or truncated:
                break
    rate = s / num_eps * 100
    print(f"  [{name}] Success rate: {rate:.0f}%")
    return rate


def random_fn(obs):
    return env.action_space.sample()


def bc_fn(obs):
    obs_vec = np.concatenate([obs["observation"], obs["achieved_goal"], obs["desired_goal"]])
    t = torch.FloatTensor(obs_vec).unsqueeze(0).to(device)
    with torch.no_grad():
        return policy(t).cpu().numpy()[0]


random_sr = evaluate(random_fn, "Random")
bc_sr = evaluate(bc_fn, "BC (learned)")
env.close()

print(f"\n  Expert: {expert_sr:.0f}% | BC: {bc_sr:.0f}% | Random: {random_sr:.0f}%")
if bc_sr < expert_sr * 0.8:
    print("  -> BC struggles on this task! We need ACT or Diffusion Policy.")
else:
    print("  -> BC works reasonably well here.")

torch.save(policy.state_dict(), "/Users/chan/robot-learning/bc_fetchpush.pt")
print("\nModel saved: bc_fetchpush.pt")
