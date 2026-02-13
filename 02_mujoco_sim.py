"""
Step 2: MuJoCo Robot Arm Simulation (Headless)
- Run a virtual robot arm with random actions
- Understand observation and action spaces
"""

import gymnasium as gym
import gymnasium_robotics
import numpy as np

gym.register_envs(gymnasium_robotics)

env = gym.make("FetchReach-v4", render_mode=None)
obs, info = env.reset()

print("=== FetchReach Environment ===")
print("Observations:")
for key, value in obs.items():
    print(f"  {key}: shape={np.array(value).shape}")
print(f"\nAction space: {env.action_space}")
print(f"  -> 4 DOF control (x, y, z movement + gripper)")

print(f"\n=== Running 100 steps with random actions ===")
total_reward = 0
for step in range(100):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    total_reward += reward

    if step % 25 == 0:
        grip_pos = obs["achieved_goal"]
        target_pos = obs["desired_goal"]
        dist = np.linalg.norm(grip_pos - target_pos)
        print(f"  Step {step:3d} | gripper->target dist: {dist:.3f} | reward: {reward:.1f}")

    if terminated or truncated:
        obs, info = env.reset()

env.close()

print(f"\nTotal reward: {total_reward:.1f}")
print("-> Random policy performs poorly. We need imitation learning!")
