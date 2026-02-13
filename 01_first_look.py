"""
Step 1: Exploring the LeRobot PushT Dataset
- Understand how imitation learning data is structured
- Examine observation and action spaces
"""

from lerobot.datasets.lerobot_dataset import LeRobotDataset
import torch

print("Downloading dataset... (may take a moment on first run)")
dataset = LeRobotDataset("lerobot/pusht", download_videos=False)

print(f"\n=== PushT Dataset Info ===")
print(f"Total frames: {len(dataset)}")
print(f"Episodes: {dataset.num_episodes}")
print(f"FPS: {dataset.fps}")

print(f"\n=== Data Keys ===")
print(f"Features: {list(dataset.features.keys())}")

print(f"\n=== Key Takeaways ===")
print("observation = what the robot 'sees' (images or state vectors)")
print("action = what the robot 'does' (joint angles, positions, etc.)")
print("imitation learning = learn to reproduce expert (observation, action) pairs!")
