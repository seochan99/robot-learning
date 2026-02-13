"""
Step 10: Head-to-Head Comparison — BC vs ACT vs Diffusion Policy

Train all three on the SAME data, evaluate on the SAME task.
This gives you intuition for when to use which algorithm.
"""

import gymnasium as gym
import gymnasium_robotics
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import time

gym.register_envs(gymnasium_robotics)

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Device: {device}")

ACTION_HORIZON = 4
OBS_DIM = 13
ACT_DIM = 4
NUM_DIFFUSION_STEPS = 20

# ============================================================
# Collect shared expert data
# ============================================================
print("Collecting expert data...")


def expert_policy(obs):
    direction = obs["desired_goal"] - obs["achieved_goal"]
    action = np.zeros(4)
    action[:3] = np.clip(direction * 10, -1, 1)
    return action


env = gym.make("FetchReach-v4", render_mode=None)
episodes = []

for ep in range(300):
    obs, _ = env.reset()
    ep_obs, ep_act = [], []
    for step in range(50):
        obs_vec = np.concatenate([obs["observation"], obs["desired_goal"]])
        action = expert_policy(obs)
        ep_obs.append(obs_vec)
        ep_act.append(action)
        obs, _, term, trunc, _ = env.step(action)
        if term or trunc:
            break
    episodes.append({"obs": np.array(ep_obs), "act": np.array(ep_act)})

env.close()

# Prepare flat data (for BC)
flat_obs, flat_act = [], []
for ep in episodes:
    flat_obs.extend(ep["obs"])
    flat_act.extend(ep["act"])
flat_obs = np.array(flat_obs)
flat_act = np.array(flat_act)

# Prepare chunked data (for ACT and Diffusion)
chunk_obs, chunk_act, chunk_obs_seq = [], [], []
OBS_HORIZON = 3
for ep in episodes:
    T = len(ep["obs"])
    for t in range(OBS_HORIZON - 1, T - ACTION_HORIZON):
        chunk_obs.append(ep["obs"][t])
        chunk_act.append(ep["act"][t : t + ACTION_HORIZON].flatten())
        obs_seq = ep["obs"][t - OBS_HORIZON + 1 : t + 1]
        chunk_obs_seq.append(obs_seq)

chunk_obs = np.array(chunk_obs)
chunk_act = np.array(chunk_act)
chunk_obs_seq = np.array(chunk_obs_seq)

print(f"Flat data: {len(flat_obs)} frames | Chunked data: {len(chunk_obs)} sequences")


# ============================================================
# Shared evaluation function
# ============================================================
def evaluate(action_fn, num_eps=100, name=""):
    env = gym.make("FetchReach-v4", render_mode=None)
    successes = 0
    for ep in range(num_eps):
        obs, _ = env.reset()
        action_queue = []
        obs_history = []
        for step in range(50):
            obs_vec = np.concatenate([obs["observation"], obs["desired_goal"]])
            obs_history.append(obs_vec)
            action, action_queue = action_fn(obs_vec, obs_history, action_queue)
            obs, _, term, trunc, info = env.step(action)
            if info.get("is_success", False):
                successes += 1
                break
            if term or trunc:
                break
    env.close()
    sr = successes / num_eps * 100
    return sr


# ============================================================
# 1. Behavioral Cloning
# ============================================================
print("\n[1/3] Training Behavioral Cloning...")


class BCPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(OBS_DIM, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, ACT_DIM), nn.Tanh(),
        )

    def forward(self, x):
        return self.net(x)


bc = BCPolicy().to(device)
bc_opt = optim.Adam(bc.parameters(), lr=1e-3)
bc_obs_t = torch.FloatTensor(flat_obs).to(device)
bc_act_t = torch.FloatTensor(flat_act).to(device)

t0 = time.time()
for epoch in range(80):
    idx = torch.randperm(len(bc_obs_t))
    for i in range(0, len(bc_obs_t), 256):
        batch = idx[i : i + 256]
        loss = nn.MSELoss()(bc(bc_obs_t[batch]), bc_act_t[batch])
        bc_opt.zero_grad()
        loss.backward()
        bc_opt.step()
bc_time = time.time() - t0
bc.eval()


def bc_action_fn(obs_vec, obs_history, action_queue):
    t = torch.FloatTensor(obs_vec).unsqueeze(0).to(device)
    with torch.no_grad():
        action = bc(t).cpu().numpy()[0]
    return action, []


# ============================================================
# 2. ACT (Action Chunking with Transformer)
# ============================================================
print("[2/3] Training ACT...")


class ACTPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        d = 256
        self.obs_enc = nn.Sequential(nn.Linear(OBS_DIM, d), nn.ReLU(), nn.Linear(d, d))
        self.pos_embed = nn.Parameter(torch.randn(1, 20, d) * 0.02)
        layer = nn.TransformerEncoderLayer(d_model=d, nhead=4, dim_feedforward=512, batch_first=True)
        self.transformer = nn.TransformerEncoder(layer, num_layers=3)
        self.head = nn.Sequential(
            nn.Linear(d, d), nn.ReLU(),
            nn.Linear(d, ACT_DIM * ACTION_HORIZON), nn.Tanh(),
        )

    def forward(self, obs_seq):
        B, T, _ = obs_seq.shape
        x = self.obs_enc(obs_seq) + self.pos_embed[:, :T, :]
        x = self.transformer(x)
        return self.head(x[:, -1, :]).view(B, ACTION_HORIZON, ACT_DIM)


act = ACTPolicy().to(device)
act_opt = optim.Adam(act.parameters(), lr=1e-4)
act_obs_t = torch.FloatTensor(chunk_obs_seq).to(device)
act_act_t = torch.FloatTensor(chunk_act).to(device).view(-1, ACTION_HORIZON, ACT_DIM)

t0 = time.time()
for epoch in range(80):
    idx = torch.randperm(len(act_obs_t))
    for i in range(0, len(act_obs_t), 256):
        batch = idx[i : i + 256]
        loss = nn.MSELoss()(act(act_obs_t[batch]), act_act_t[batch])
        act_opt.zero_grad()
        loss.backward()
        act_opt.step()
act_time = time.time() - t0
act.eval()


def act_action_fn(obs_vec, obs_history, action_queue):
    if len(action_queue) > 0:
        return action_queue.pop(0), action_queue

    if len(obs_history) >= OBS_HORIZON:
        recent = np.array(obs_history[-OBS_HORIZON:])
    else:
        pad = [obs_history[0]] * (OBS_HORIZON - len(obs_history))
        recent = np.array(pad + list(obs_history))

    t = torch.FloatTensor(recent).unsqueeze(0).to(device)
    with torch.no_grad():
        chunk = act(t).cpu().numpy()[0]
    queue = list(chunk[1:])
    return chunk[0], queue


# ============================================================
# 3. Diffusion Policy
# ============================================================
print("[3/3] Training Diffusion Policy...")

betas = torch.linspace(1e-4, 0.02, NUM_DIFFUSION_STEPS).to(device)
alphas = 1.0 - betas
alphas_cumprod = torch.cumprod(alphas, dim=0)
sqrt_ac = torch.sqrt(alphas_cumprod)
sqrt_omc = torch.sqrt(1.0 - alphas_cumprod)


class NoisePredictor(nn.Module):
    def __init__(self):
        super().__init__()
        h = 256
        act_flat = ACT_DIM * ACTION_HORIZON
        self.t_emb = nn.Sequential(nn.Linear(1, h), nn.SiLU(), nn.Linear(h, h))
        self.o_emb = nn.Sequential(nn.Linear(OBS_DIM, h), nn.SiLU(), nn.Linear(h, h))
        self.net = nn.Sequential(
            nn.Linear(act_flat + h + h, h * 2), nn.SiLU(),
            nn.Linear(h * 2, h * 2), nn.SiLU(),
            nn.Linear(h * 2, h), nn.SiLU(),
            nn.Linear(h, act_flat),
        )

    def forward(self, noisy_act, obs, t):
        return self.net(torch.cat([noisy_act, self.o_emb(obs), self.t_emb(t)], -1))


diff = NoisePredictor().to(device)
diff_opt = optim.Adam(diff.parameters(), lr=1e-4)
diff_obs_t = torch.FloatTensor(chunk_obs).to(device)
diff_act_t = torch.FloatTensor(chunk_act).to(device)

t0 = time.time()
for epoch in range(120):
    idx = torch.randperm(len(diff_obs_t))
    for i in range(0, len(diff_obs_t), 256):
        batch = idx[i : i + 256]
        B = len(batch)
        t_step = torch.randint(0, NUM_DIFFUSION_STEPS, (B,), device=device)
        t_norm = t_step.float().unsqueeze(-1) / NUM_DIFFUSION_STEPS
        noise = torch.randn_like(diff_act_t[batch])
        noisy = sqrt_ac[t_step].unsqueeze(-1) * diff_act_t[batch] + sqrt_omc[t_step].unsqueeze(-1) * noise
        loss = nn.MSELoss()(diff(noisy, diff_obs_t[batch], t_norm), noise)
        diff_opt.zero_grad()
        loss.backward()
        diff_opt.step()
diff_time = time.time() - t0
diff.eval()


@torch.no_grad()
def denoise(obs_vec):
    obs_t = torch.FloatTensor(obs_vec).unsqueeze(0).to(device)
    x = torch.randn(1, ACTION_HORIZON * ACT_DIM, device=device)
    for s in reversed(range(NUM_DIFFUSION_STEPS)):
        t = torch.tensor([[s / NUM_DIFFUSION_STEPS]], device=device)
        pred = diff(x, obs_t, t)
        a, ac, b = alphas[s], alphas_cumprod[s], betas[s]
        z = torch.randn_like(x) if s > 0 else torch.zeros_like(x)
        x = (1 / a.sqrt()) * (x - (b / (1 - ac).sqrt()) * pred) + b.sqrt() * z
    return np.clip(x.cpu().numpy()[0].reshape(ACTION_HORIZON, ACT_DIM), -1, 1)


def diff_action_fn(obs_vec, obs_history, action_queue):
    if len(action_queue) > 0:
        return action_queue.pop(0), action_queue
    chunk = denoise(obs_vec)
    queue = list(chunk[1:])
    return chunk[0], queue


# ============================================================
# Head-to-head evaluation
# ============================================================
print("\n" + "=" * 60)
print("  HEAD-TO-HEAD COMPARISON")
print("=" * 60)

bc_sr = evaluate(bc_action_fn, name="BC")
act_sr = evaluate(act_action_fn, name="ACT")
diff_sr = evaluate(diff_action_fn, name="Diffusion")

print(f"\n{'Method':<20} {'Success Rate':>14} {'Train Time':>12}")
print("-" * 50)
print(f"{'Behavioral Cloning':<20} {bc_sr:>13.0f}% {bc_time:>10.1f}s")
print(f"{'ACT':<20} {act_sr:>13.0f}% {act_time:>10.1f}s")
print(f"{'Diffusion Policy':<20} {diff_sr:>13.0f}% {diff_time:>10.1f}s")

print(f"\n{'='*50}")
print("Key Takeaways:")
print(f"  BC:        Simple, fast, but single-step prediction")
print(f"  ACT:       Action chunking = smoother trajectories")
print(f"  Diffusion: Iterative denoising = handles multimodality")
print(f"\nFor harder tasks (FetchPush, real robots), the gap widens.")
print(f"Next step: try these on GPU with LeRobot's full implementations!")
