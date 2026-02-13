"""
Step 4: Imitation Learning from Scratch

Full pipeline:
1. Collect expert demonstration data
2. Train a neural network to imitate the expert
3. Evaluate: learned policy vs random baseline

This is the core loop of behavioral cloning (BC),
the simplest form of imitation learning.
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
# Step 1: Collect expert demonstrations
# ============================================================
print("\n" + "=" * 50)
print("Step 1: Collecting expert demonstrations")
print("=" * 50)


def expert_policy(obs):
    """Simple expert: move directly toward the target."""
    grip_pos = obs["achieved_goal"]
    target_pos = obs["desired_goal"]
    direction = target_pos - grip_pos
    action = np.zeros(4)
    action[:3] = np.clip(direction * 10, -1, 1)
    return action


env = gym.make("FetchReach-v4", render_mode=None)
expert_data = defaultdict(list)
num_episodes = 200

for ep in range(num_episodes):
    obs, info = env.reset()

    for step in range(50):
        obs_vector = np.concatenate([obs["observation"], obs["desired_goal"]])
        action = expert_policy(obs)

        expert_data["observations"].append(obs_vector)
        expert_data["actions"].append(action)

        obs, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            break

env.close()

obs_array = np.array(expert_data["observations"])
act_array = np.array(expert_data["actions"])
print(f"Collected {len(obs_array)} frames")
print(f"  Observation dim: {obs_array.shape[1]}")
print(f"  Action dim: {act_array.shape[1]}")

# ============================================================
# Step 2: Train imitation learning policy
# ============================================================
print("\n" + "=" * 50)
print("Step 2: Training neural network (behavioral cloning)")
print("=" * 50)


class ImitationPolicy(nn.Module):
    """Simple MLP that maps observations to actions."""

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


obs_tensor = torch.FloatTensor(obs_array).to(device)
act_tensor = torch.FloatTensor(act_array).to(device)

obs_dim = obs_array.shape[1]
act_dim = act_array.shape[1]
policy = ImitationPolicy(obs_dim, act_dim).to(device)
optimizer = optim.Adam(policy.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

batch_size = 256
num_epochs = 50
dataset_size = len(obs_tensor)

for epoch in range(num_epochs):
    indices = torch.randperm(dataset_size)
    total_loss = 0
    num_batches = 0

    for i in range(0, dataset_size, batch_size):
        batch_idx = indices[i : i + batch_size]
        batch_obs = obs_tensor[batch_idx]
        batch_act = act_tensor[batch_idx]

        pred_act = policy(batch_obs)
        loss = loss_fn(pred_act, batch_act)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    if (epoch + 1) % 10 == 0:
        avg_loss = total_loss / num_batches
        print(f"  Epoch {epoch + 1:3d}/{num_epochs} | Loss: {avg_loss:.6f}")

# ============================================================
# Step 3: Evaluate learned policy vs random baseline
# ============================================================
print("\n" + "=" * 50)
print("Step 3: Evaluation")
print("=" * 50)


def evaluate_policy(env, policy_fn, num_episodes=50, name=""):
    successes = 0
    total_rewards = []

    for ep in range(num_episodes):
        obs, info = env.reset()
        ep_reward = 0

        for step in range(50):
            action = policy_fn(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward

            if info.get("is_success", False):
                successes += 1
                break

            if terminated or truncated:
                break

        total_rewards.append(ep_reward)

    success_rate = successes / num_episodes * 100
    avg_reward = np.mean(total_rewards)
    print(f"  [{name}] Success rate: {success_rate:.0f}% | Avg reward: {avg_reward:.1f}")
    return success_rate


env = gym.make("FetchReach-v4", render_mode=None)


def random_policy(obs):
    return env.action_space.sample()


policy.eval()


def learned_policy(obs):
    obs_vector = np.concatenate([obs["observation"], obs["desired_goal"]])
    obs_t = torch.FloatTensor(obs_vector).unsqueeze(0).to(device)
    with torch.no_grad():
        action = policy(obs_t).cpu().numpy()[0]
    return action


print("Evaluating...")
random_sr = evaluate_policy(env, random_policy, name="Random")
learned_sr = evaluate_policy(env, learned_policy, name="Learned")

env.close()

print("\n" + "=" * 50)
print("Results")
print("=" * 50)
print(f"  Random policy:  {random_sr:.0f}%")
print(f"  Learned policy: {learned_sr:.0f}%")
print(f"\n  -> Imitation learning outperforms random by {learned_sr - random_sr:.0f}%p!")

model_path = "/Users/chan/robot-learning/trained_policy.pt"
torch.save(policy.state_dict(), model_path)
print(f"\nModel saved: {model_path}")
print("Run 05_visualize_learned.py to see the trained policy in action!")
