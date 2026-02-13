"""
Step 11: Teleoperation — Control the Robot Yourself!

YOU control the robot arm with keyboard, collect demonstration data,
then train a policy to imitate YOUR behavior.

Controls (press in the TERMINAL, not the MuJoCo window):
  W/S  — forward / backward (Y axis)
  A/D  — left / right (X axis)
  Q/E  — up / down (Z axis)
  G    — toggle gripper (open/close)
  R    — start/stop recording an episode
  T    — train policy on collected data
  V    — switch to autonomous mode (watch learned policy)
  M    — switch back to manual control
  X    — quit

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
import sys
import os
import select
import tty
import termios
import time

gym.register_envs(gymnasium_robotics)

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# ============================================================
# Non-blocking keyboard input (no pynput, no crash)
# ============================================================

old_settings = termios.tcgetattr(sys.stdin)


def init_keyboard():
    tty.setcbreak(sys.stdin.fileno())


def restore_keyboard():
    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)


def get_key():
    """Non-blocking key read. Returns key char or None."""
    if select.select([sys.stdin], [], [], 0)[0]:
        return sys.stdin.read(1).lower()
    return None


# ============================================================
# Policy network
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
SPEED = 1.0

env = gym.make(ENV_NAME, render_mode="human", max_episode_steps=200)
obs, info = env.reset()
env.render()

all_episodes = []
current_episode = {"obs": [], "act": []}
recording = False
auto_mode = False
gripper_closed = False
policy = None
held_keys = set()

print("=" * 55)
print("  TELEOPERATION MODE — FetchPush")
print("=" * 55)
print("  W/S = forward/back  |  A/D = left/right")
print("  Q/E = up/down       |  G   = toggle gripper")
print("  R   = record        |  T   = train")
print("  V   = auto mode     |  M   = manual mode")
print("  X   = quit")
print("=" * 55)
print("  Press R to start recording!")
print()

init_keyboard()

try:
    while True:
        # Read all available keys
        key = get_key()

        if key == "x":
            break

        if key == "r":
            if not recording:
                recording = True
                current_episode = {"obs": [], "act": []}
                obs, info = env.reset()
                env.render()
                print("\r  [REC] Recording! Control the robot.        ")
            else:
                recording = False
                if len(current_episode["obs"]) > 10:
                    all_episodes.append(current_episode)
                    print(f"\r  [SAVE] Episode saved! ({len(current_episode['obs'])} frames, {len(all_episodes)} total)        ")
                else:
                    print("\r  [SKIP] Too short, discarded.        ")
                current_episode = {"obs": [], "act": []}
                obs, info = env.reset()
                env.render()

        if key == "t":
            if len(all_episodes) < 3:
                print(f"\r  [!] Need 3+ episodes. Have {len(all_episodes)}.        ")
            else:
                print(f"\r  [TRAIN] Training on {len(all_episodes)} episodes...        ")
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
                torch.save(policy.state_dict(), "my_teleop_policy.pt")
                print(f"\r  [DONE] Trained on {len(train_obs)} frames! Press V to watch.        ")

        if key == "v":
            if policy is not None:
                auto_mode = True
                obs, info = env.reset()
                env.render()
                print("\r  [AUTO] Watching learned policy. Press M for manual.        ")
            else:
                print("\r  [!] Train first (press T).        ")

        if key == "m":
            auto_mode = False
            obs, info = env.reset()
            env.render()
            print("\r  [MANUAL] You're in control.        ")

        if key == "g":
            gripper_closed = not gripper_closed
            state = "CLOSED" if gripper_closed else "OPEN"
            print(f"\r  [GRIPPER] {state}        ")

        # Track held keys
        if key in ("w", "s", "a", "d", "q", "e"):
            held_keys.add(key)
        # Release keys after a short hold (simple approach)
        release_keys = set()
        for k in held_keys:
            if key is None or key != k:
                release_keys.add(k)
        # Keep key active for a few frames by not clearing immediately

        # Generate action
        if auto_mode and policy is not None:
            obs_vec = np.concatenate([obs["observation"], obs["achieved_goal"], obs["desired_goal"]])
            t = torch.FloatTensor(obs_vec).unsqueeze(0).to(device)
            with torch.no_grad():
                action = policy(t).cpu().numpy()[0]
        else:
            action = np.zeros(4)
            if key == "w" or "w" in held_keys: action[1] += SPEED
            if key == "s" or "s" in held_keys: action[1] -= SPEED
            if key == "d" or "d" in held_keys: action[0] += SPEED
            if key == "a" or "a" in held_keys: action[0] -= SPEED
            if key == "e" or "e" in held_keys: action[2] += SPEED
            if key == "q" or "q" in held_keys: action[2] -= SPEED
            action[3] = -1.0 if gripper_closed else 1.0

            # Clear held keys after use
            held_keys.clear()

        # Record
        if recording and not auto_mode:
            obs_vec = np.concatenate([obs["observation"], obs["achieved_goal"], obs["desired_goal"]])
            current_episode["obs"].append(obs_vec)
            current_episode["act"].append(action.copy())

        # Step
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()

        if info.get("is_success", False):
            if recording:
                print("\r  [SUCCESS!] Press R to save this episode.        ")
            elif auto_mode:
                print("\r  [SUCCESS!] Policy did it!        ")

        if terminated or truncated:
            obs, info = env.reset()
            env.render()

        time.sleep(0.03)

except Exception as e:
    print(f"\nError: {e}")
finally:
    restore_keyboard()
    env.close()
    if all_episodes:
        print(f"\nCollected {len(all_episodes)} episodes total.")
    print("Done!")
