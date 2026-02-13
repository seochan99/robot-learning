"""
Visualize any trained policy in MuJoCo.

Usage:
  python visualize.py random        # random actions (baseline)
  python visualize.py expert        # heuristic expert
  python visualize.py bc            # behavioral cloning (step 4)
  python visualize.py act           # ACT policy (step 8)
  python visualize.py diffusion     # diffusion policy (step 9)
  python visualize.py push          # BC on FetchPush (step 6)

Press Ctrl+C to exit.
"""

import sys
import gymnasium as gym
import gymnasium_robotics
import numpy as np
import torch
import torch.nn as nn
import time

gym.register_envs(gymnasium_robotics)

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

ACTION_HORIZON = 4
NUM_DIFFUSION_STEPS = 20
OBS_HORIZON = 3

# ============================================================
# Model definitions (must match training)
# ============================================================

class BCPolicy(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, act_dim), nn.Tanh(),
        )
    def forward(self, x):
        return self.net(x)


class BCPushPolicy(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, act_dim), nn.Tanh(),
        )
    def forward(self, x):
        return self.net(x)


class ACTPolicy(nn.Module):
    def __init__(self, obs_dim=13, act_dim=4, chunk_size=5, d=256):
        super().__init__()
        self.chunk_size = chunk_size
        self.act_dim = act_dim
        self.obs_encoder = nn.Sequential(nn.Linear(obs_dim, d), nn.ReLU(), nn.Linear(d, d))
        self.pos_embed = nn.Parameter(torch.randn(1, 20, d) * 0.02)
        layer = nn.TransformerEncoderLayer(d_model=d, nhead=4, dim_feedforward=512, batch_first=True)
        self.transformer = nn.TransformerEncoder(layer, num_layers=3)
        self.action_head = nn.Sequential(
            nn.Linear(d, d), nn.ReLU(),
            nn.Linear(d, act_dim * chunk_size), nn.Tanh(),
        )
    def forward(self, obs_seq):
        B, T, _ = obs_seq.shape
        x = self.obs_encoder(obs_seq) + self.pos_embed[:, :T, :]
        x = self.transformer(x)
        return self.action_head(x[:, -1, :]).view(B, self.chunk_size, self.act_dim)


class NoisePredictor(nn.Module):
    def __init__(self, obs_dim=13, act_dim=4, action_horizon=4, h=256):
        super().__init__()
        act_flat = act_dim * action_horizon
        self.time_embed = nn.Sequential(nn.Linear(1, h), nn.SiLU(), nn.Linear(h, h))
        self.obs_encoder = nn.Sequential(nn.Linear(obs_dim, h), nn.SiLU(), nn.Linear(h, h))
        self.net = nn.Sequential(
            nn.Linear(act_flat + h + h, h * 2), nn.SiLU(),
            nn.Linear(h * 2, h * 2), nn.SiLU(),
            nn.Linear(h * 2, h), nn.SiLU(),
            nn.Linear(h, act_flat),
        )
    def forward(self, noisy_act, obs, t):
        return self.net(torch.cat([noisy_act, self.obs_encoder(obs), self.time_embed(t)], -1))


# ============================================================
# Diffusion sampling
# ============================================================
betas = torch.linspace(1e-4, 0.02, NUM_DIFFUSION_STEPS).to(device)
alphas = 1.0 - betas
alphas_cumprod = torch.cumprod(alphas, dim=0)


@torch.no_grad()
def diffusion_sample(model, obs_vec):
    obs_t = torch.FloatTensor(obs_vec).unsqueeze(0).to(device)
    x = torch.randn(1, ACTION_HORIZON * 4, device=device)
    for s in reversed(range(NUM_DIFFUSION_STEPS)):
        t = torch.tensor([[s / NUM_DIFFUSION_STEPS]], device=device)
        pred = model(x, obs_t, t)
        a, ac, b = alphas[s], alphas_cumprod[s], betas[s]
        z = torch.randn_like(x) if s > 0 else torch.zeros_like(x)
        x = (1 / a.sqrt()) * (x - (b / (1 - ac).sqrt()) * pred) + b.sqrt() * z
    return np.clip(x.cpu().numpy()[0].reshape(ACTION_HORIZON, 4), -1, 1)


# ============================================================
# Policy loaders
# ============================================================

def load_policy(mode):
    if mode == "random":
        return "FetchReach-v4", None

    elif mode == "expert":
        return "FetchReach-v4", None

    elif mode == "bc":
        model = BCPolicy(13, 4).to(device)
        model.load_state_dict(torch.load("trained_policy.pt", weights_only=True, map_location=device))
        model.eval()
        return "FetchReach-v4", model

    elif mode == "push":
        model = BCPushPolicy(31, 4).to(device)
        model.load_state_dict(torch.load("bc_fetchpush.pt", weights_only=True, map_location=device))
        model.eval()
        return "FetchPush-v4", model

    elif mode == "act":
        model = ACTPolicy().to(device)
        model.load_state_dict(torch.load("act_fetchreach.pt", weights_only=True, map_location=device))
        model.eval()
        return "FetchReach-v4", model

    elif mode == "diffusion":
        model = NoisePredictor().to(device)
        model.load_state_dict(torch.load("diffusion_fetchreach.pt", weights_only=True, map_location=device))
        model.eval()
        return "FetchReach-v4", model

    else:
        print(f"Unknown mode: {mode}")
        print("Options: random, expert, bc, act, diffusion, push")
        sys.exit(1)


def get_action(mode, model, obs, obs_history, action_queue):
    if mode == "random":
        return env.action_space.sample(), []

    elif mode == "expert":
        direction = obs["desired_goal"] - obs["achieved_goal"]
        action = np.zeros(4)
        action[:3] = np.clip(direction * 10, -1, 1)
        return action, []

    elif mode == "bc":
        obs_vec = np.concatenate([obs["observation"], obs["desired_goal"]])
        t = torch.FloatTensor(obs_vec).unsqueeze(0).to(device)
        with torch.no_grad():
            return model(t).cpu().numpy()[0], []

    elif mode == "push":
        obs_vec = np.concatenate([obs["observation"], obs["achieved_goal"], obs["desired_goal"]])
        t = torch.FloatTensor(obs_vec).unsqueeze(0).to(device)
        with torch.no_grad():
            return model(t).cpu().numpy()[0], []

    elif mode == "act":
        if len(action_queue) > 0:
            return action_queue.pop(0), action_queue

        if len(obs_history) >= OBS_HORIZON:
            recent = np.array(obs_history[-OBS_HORIZON:])
        else:
            pad = [obs_history[0]] * (OBS_HORIZON - len(obs_history))
            recent = np.array(pad + list(obs_history))

        t = torch.FloatTensor(recent).unsqueeze(0).to(device)
        with torch.no_grad():
            chunk = model(t).cpu().numpy()[0]
        return chunk[0], list(chunk[1:])

    elif mode == "diffusion":
        if len(action_queue) > 0:
            return action_queue.pop(0), action_queue

        obs_vec = np.concatenate([obs["observation"], obs["desired_goal"]])
        chunk = diffusion_sample(model, obs_vec)
        return chunk[0], list(chunk[1:])


# ============================================================
# Main loop
# ============================================================
if len(sys.argv) < 2:
    print(__doc__)
    sys.exit(0)

mode = sys.argv[1].lower()
env_name, model = load_policy(mode)

LABELS = {
    "random": "RANDOM POLICY (baseline)",
    "expert": "HEURISTIC EXPERT",
    "bc": "BEHAVIORAL CLONING (step 4)",
    "push": "BC ON FETCH-PUSH (step 6)",
    "act": "ACT - ACTION CHUNKING TRANSFORMER (step 8)",
    "diffusion": "DIFFUSION POLICY (step 9)",
}

env = gym.make(env_name, render_mode="human", max_episode_steps=100)

print("=" * 55)
print(f"  {LABELS.get(mode, mode)}")
print(f"  Environment: {env_name}")
print(f"  Exit: Ctrl+C")
print("=" * 55)

try:
    episode = 0
    total_success = 0

    while True:
        obs, info = env.reset()
        env.render()
        episode += 1
        success = False
        action_queue = []
        obs_history = []

        for step in range(100):
            obs_vec = np.concatenate([obs["observation"], obs["desired_goal"]])
            obs_history.append(obs_vec)

            action, action_queue = get_action(mode, model, obs, obs_history, action_queue)
            obs, reward, terminated, truncated, info = env.step(action)
            env.render()
            time.sleep(0.02)

            if info.get("is_success", False):
                success = True
                total_success += 1
                print(f"  Episode {episode}: SUCCESS ({step + 1} steps)  [{total_success}/{episode} total]")
                time.sleep(0.3)
                break

            if terminated or truncated:
                break

        if not success:
            print(f"  Episode {episode}: FAILED  [{total_success}/{episode} total]")

except KeyboardInterrupt:
    rate = total_success / max(episode, 1) * 100
    print(f"\n  Final: {total_success}/{episode} ({rate:.0f}%)")
    print("Exiting!")
finally:
    env.close()
