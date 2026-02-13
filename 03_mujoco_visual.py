"""
Step 3: MuJoCo Visual Rendering
- Opens a window to watch the robot arm move in real-time
- Mouse controls: left-drag=rotate, right-drag=pan, scroll=zoom
- Press Ctrl+C to exit
"""

import gymnasium as gym
import gymnasium_robotics
import numpy as np
import time

gym.register_envs(gymnasium_robotics)

print("Opening MuJoCo viewer...")

env = gym.make("FetchReach-v4", render_mode="human", max_episode_steps=200)
obs, info = env.reset()
env.render()

print("MuJoCo window is open! Robot arm is moving randomly.")
print("Press Ctrl+C to exit.")

try:
    step = 0
    while True:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()
        time.sleep(0.02)

        if terminated or truncated:
            obs, info = env.reset()
            step = 0

        step += 1

except KeyboardInterrupt:
    print("\nExiting!")
finally:
    env.close()
