"""
Step 5: Visualize Trained Imitation Learning Policy in MuJoCo
- Watch the robot arm reach targets using the learned policy
- Run 04_imitation_learning_demo.py first to generate trained_policy.pt
- Press Ctrl+C to exit
"""

import gymnasium as gym
import gymnasium_robotics
import numpy as np
import torch
import torch.nn as nn
import time

gym.register_envs(gymnasium_robotics)

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


class ImitationPolicy(nn.Module):
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


policy = ImitationPolicy(obs_dim=13, act_dim=4).to(device)
policy.load_state_dict(
    torch.load("/Users/chan/robot-learning/trained_policy.pt", weights_only=True)
)
policy.eval()


def learned_policy(obs):
    obs_vector = np.concatenate([obs["observation"], obs["desired_goal"]])
    obs_t = torch.FloatTensor(obs_vector).unsqueeze(0).to(device)
    with torch.no_grad():
        action = policy(obs_t).cpu().numpy()[0]
    return action


env = gym.make("FetchReach-v4", render_mode="human", max_episode_steps=100)

print("=" * 50)
print(" Trained Imitation Learning Policy Visualization")
print(" Robot arm reaches toward the red target!")
print(" Exit: Ctrl+C")
print("=" * 50)

try:
    episode = 0
    while True:
        obs, info = env.reset()
        env.render()
        episode += 1
        success = False

        for step in range(100):
            action = learned_policy(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            env.render()
            time.sleep(0.02)

            if info.get("is_success", False):
                success = True
                print(f"  Episode {episode}: SUCCESS ({step + 1} steps)")
                time.sleep(0.5)
                break

            if terminated or truncated:
                break

        if not success:
            print(f"  Episode {episode}: FAILED")

except KeyboardInterrupt:
    print("\nExiting!")
finally:
    env.close()
