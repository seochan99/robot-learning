"""
Step 11: Teleoperation — Control the Robot Yourself!

YOU control the robot arm with keyboard, collect demonstration data,
then train a policy to imitate YOUR behavior.

This is exactly how real-world imitation learning works:
1. Human teleoperates the robot
2. Record (observation, action) pairs
3. Train a neural network to reproduce the behavior

Controls:
  W/S  — move forward / backward (Y axis)
  A/D  — move left / right (X axis)
  Q/E  — move up / down (Z axis)
  G    — toggle gripper (open/close)
  R    — start/stop recording an episode
  T    — train policy on collected data
  V    — switch to autonomous mode (watch learned policy)
  M    — switch back to manual control
  ESC  — quit

Run:
  cd ~/robot-learning && source .venv/bin/activate
  python 11_teleop_and_learn.py
"""

import gymnasium as gym
import gymnasium_robotics
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import time
import threading
from collections import defaultdict

gym.register_envs(gymnasium_robotics)

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# ============================================================
# Keyboard input handler (runs in background thread)
# ============================================================
keys_pressed = set()
lock = threading.Lock()


def start_keyboard_listener():
    from pynput import keyboard

    def on_press(key):
        with lock:
            try:
                keys_pressed.add(key.char.lower())
            except AttributeError:
                if key == keyboard.Key.esc:
                    keys_pressed.add("esc")
                elif key == keyboard.Key.space:
                    keys_pressed.add("space")

    def on_release(key):
        with lock:
            try:
                keys_pressed.discard(key.char.lower())
            except AttributeError:
                if key == keyboard.Key.esc:
                    keys_pressed.discard("esc")
                elif key == keyboard.Key.space:
                    keys_pressed.discard("space")

    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.daemon = True
    listener.start()
    return listener


# ============================================================
# Policy network (same as before)
# ============================================================
class ImitationPolicy(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, act_dim), nn.Tanh(),
        )

    def forward(self, x):
        return self.net(x)


# ============================================================
# Main
# ============================================================
ENV_NAME = "FetchPush-v4"
SPEED = 0.8
gripper_closed = False

env = gym.make(ENV_NAME, render_mode="human", max_episode_steps=200)
obs, info = env.reset()
env.render()

# Data storage
all_episodes = []
current_episode = {"obs": [], "act": []}
recording = False
auto_mode = False
policy = None

listener = start_keyboard_listener()

print("=" * 55)
print("  TELEOPERATION MODE")
print("=" * 55)
print("  W/S = forward/back  |  A/D = left/right")
print("  Q/E = up/down       |  G   = toggle gripper")
print("  R   = record        |  T   = train on your data")
print("  V   = auto mode     |  M   = manual mode")
print("  ESC = quit")
print("=" * 55)
print("  Press R to start recording an episode!")
print()

try:
    step_count = 0
    while True:
        with lock:
            current_keys = keys_pressed.copy()

        if "esc" in current_keys:
            break

        # Toggle recording
        if "r" in current_keys:
            keys_pressed.discard("r")
            if not recording:
                recording = True
                current_episode = {"obs": [], "act": []}
                obs, info = env.reset()
                env.render()
                print("  [REC] Recording started! Control the robot.")
            else:
                recording = False
                if len(current_episode["obs"]) > 5:
                    all_episodes.append(current_episode)
                    print(f"  [SAVE] Episode saved! ({len(current_episode['obs'])} frames, {len(all_episodes)} total episodes)")
                else:
                    print("  [SKIP] Episode too short, discarded.")
                current_episode = {"obs": [], "act": []}
                obs, info = env.reset()
                env.render()

        # Train on collected data
        if "t" in current_keys:
            keys_pressed.discard("t")
            if len(all_episodes) < 3:
                print(f"  [!] Need at least 3 episodes. You have {len(all_episodes)}.")
            else:
                print(f"  [TRAIN] Training on {len(all_episodes)} episodes...")
                # Flatten all episodes
                train_obs, train_act = [], []
                for ep in all_episodes:
                    train_obs.extend(ep["obs"])
                    train_act.extend(ep["act"])

                obs_arr = np.array(train_obs)
                act_arr = np.array(train_act)
                obs_t = torch.FloatTensor(obs_arr).to(device)
                act_t = torch.FloatTensor(act_arr).to(device)

                policy = ImitationPolicy(obs_arr.shape[1], act_arr.shape[1]).to(device)
                optimizer = optim.Adam(policy.parameters(), lr=1e-3)

                for epoch in range(100):
                    idx = torch.randperm(len(obs_t))
                    for i in range(0, len(obs_t), 128):
                        batch = idx[i : i + 128]
                        loss = nn.MSELoss()(policy(obs_t[batch]), act_t[batch])
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                policy.eval()
                print(f"  [DONE] Trained on {len(train_obs)} frames! Press V to watch.")
                torch.save(policy.state_dict(), "my_teleop_policy.pt")

        # Toggle auto mode
        if "v" in current_keys:
            keys_pressed.discard("v")
            if policy is not None:
                auto_mode = True
                obs, info = env.reset()
                env.render()
                print("  [AUTO] Watching learned policy... Press M for manual.")
            else:
                print("  [!] Train first! Press T after recording episodes.")

        if "m" in current_keys:
            keys_pressed.discard("m")
            auto_mode = False
            obs, info = env.reset()
            env.render()
            print("  [MANUAL] You're in control again.")

        # Toggle gripper
        if "g" in current_keys:
            keys_pressed.discard("g")
            gripper_closed = not gripper_closed

        # Generate action
        if auto_mode and policy is not None:
            obs_vec = np.concatenate([obs["observation"], obs["achieved_goal"], obs["desired_goal"]])
            t = torch.FloatTensor(obs_vec).unsqueeze(0).to(device)
            with torch.no_grad():
                action = policy(t).cpu().numpy()[0]
        else:
            action = np.zeros(4)
            if "w" in current_keys: action[1] += SPEED
            if "s" in current_keys: action[1] -= SPEED
            if "d" in current_keys: action[0] += SPEED
            if "a" in current_keys: action[0] -= SPEED
            if "e" in current_keys: action[2] += SPEED
            if "q" in current_keys: action[2] -= SPEED
            action[3] = -1.0 if gripper_closed else 1.0

        # Record if active
        if recording and not auto_mode:
            obs_vec = np.concatenate([obs["observation"], obs["achieved_goal"], obs["desired_goal"]])
            current_episode["obs"].append(obs_vec)
            current_episode["act"].append(action.copy())

        # Step environment
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()

        if info.get("is_success", False):
            if recording:
                print("  [SUCCESS!] Great job! Press R to stop recording.")
            elif auto_mode:
                print("  [SUCCESS!] Policy did it!")

        if terminated or truncated:
            if auto_mode:
                obs, info = env.reset()
                env.render()

        time.sleep(0.02)
        step_count += 1

except KeyboardInterrupt:
    pass
finally:
    env.close()
    if len(all_episodes) > 0:
        print(f"\nCollected {len(all_episodes)} episodes total.")
    print("Done!")
