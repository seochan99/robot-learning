"""
Step 9: Diffusion Policy — Generate Actions from Noise

The core algorithm behind many state-of-the-art robot systems.

Key idea: treat action generation as a DENOISING process.
1. Start with pure noise
2. Iteratively denoise it into a valid action sequence
3. The model learns "what good actions look like" from expert data

This handles multimodal distributions better than BC.
e.g., if an object can be pushed from the left OR the right,
BC averages both and pushes from the middle (fails!).
Diffusion can sample either mode correctly.
"""

import gymnasium as gym
import gymnasium_robotics
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import math

gym.register_envs(gymnasium_robotics)

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Device: {device}")

ACTION_HORIZON = 4  # predict 4 future actions
OBS_DIM = 13
ACT_DIM = 4
NUM_DIFFUSION_STEPS = 20


# ============================================================
# Noise schedule (linear beta schedule)
# ============================================================


def linear_beta_schedule(timesteps, beta_start=1e-4, beta_end=0.02):
    return torch.linspace(beta_start, beta_end, timesteps)


betas = linear_beta_schedule(NUM_DIFFUSION_STEPS).to(device)
alphas = 1.0 - betas
alphas_cumprod = torch.cumprod(alphas, dim=0)
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)


# ============================================================
# Noise prediction network (conditional on observation)
# ============================================================


class NoisePredictor(nn.Module):
    """
    Given: noisy action chunk + observation + diffusion timestep
    Predict: the noise that was added
    """

    def __init__(self, obs_dim, act_dim, action_horizon, hidden=256):
        super().__init__()
        self.action_horizon = action_horizon
        self.act_dim = act_dim
        act_flat = act_dim * action_horizon

        # Timestep embedding
        self.time_embed = nn.Sequential(
            nn.Linear(1, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
        )

        # Observation encoder
        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
        )

        # Noise prediction
        self.net = nn.Sequential(
            nn.Linear(act_flat + hidden + hidden, hidden * 2),
            nn.SiLU(),
            nn.Linear(hidden * 2, hidden * 2),
            nn.SiLU(),
            nn.Linear(hidden * 2, hidden),
            nn.SiLU(),
            nn.Linear(hidden, act_flat),
        )

    def forward(self, noisy_actions, obs, timestep):
        """
        noisy_actions: (B, action_horizon * act_dim)
        obs: (B, obs_dim)
        timestep: (B, 1) normalized to [0, 1]
        """
        t_emb = self.time_embed(timestep)
        o_emb = self.obs_encoder(obs)
        x = torch.cat([noisy_actions, o_emb, t_emb], dim=-1)
        return self.net(x)


# ============================================================
# Collect expert data
# ============================================================
print("\n" + "=" * 50)
print("Step 1: Collecting expert data")
print("=" * 50)


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

# Build (obs, action_chunk) pairs
obs_list, act_list = [], []
for ep in episodes:
    T = len(ep["obs"])
    for t in range(T - ACTION_HORIZON):
        obs_list.append(ep["obs"][t])
        act_list.append(ep["act"][t : t + ACTION_HORIZON].flatten())

obs_data = torch.FloatTensor(np.array(obs_list)).to(device)
act_data = torch.FloatTensor(np.array(act_list)).to(device)
print(f"Training pairs: {len(obs_data)}")

# ============================================================
# Train diffusion policy
# ============================================================
print("\n" + "=" * 50)
print("Step 2: Training diffusion policy")
print("=" * 50)

model = NoisePredictor(OBS_DIM, ACT_DIM, ACTION_HORIZON).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-4)
N = len(obs_data)

for epoch in range(150):
    idx = torch.randperm(N)
    total_loss = 0
    n_batch = 0

    for i in range(0, N, 256):
        batch = idx[i : i + 256]
        B = len(batch)

        # Sample random timesteps
        t = torch.randint(0, NUM_DIFFUSION_STEPS, (B,), device=device)
        t_normalized = t.float().unsqueeze(-1) / NUM_DIFFUSION_STEPS

        # Add noise to actions (forward diffusion)
        noise = torch.randn_like(act_data[batch])
        sqrt_alpha = sqrt_alphas_cumprod[t].unsqueeze(-1)
        sqrt_one_minus = sqrt_one_minus_alphas_cumprod[t].unsqueeze(-1)
        noisy_actions = sqrt_alpha * act_data[batch] + sqrt_one_minus * noise

        # Predict noise
        pred_noise = model(noisy_actions, obs_data[batch], t_normalized)
        loss = nn.MSELoss()(pred_noise, noise)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batch += 1

    if (epoch + 1) % 30 == 0:
        print(f"  Epoch {epoch + 1:3d}/150 | Loss: {total_loss / n_batch:.6f}")


# ============================================================
# Inference: denoise random noise into actions (DDPM sampling)
# ============================================================
@torch.no_grad()
def sample_actions(obs_vec):
    """Generate action chunk from observation via iterative denoising."""
    model.eval()
    obs_t = torch.FloatTensor(obs_vec).unsqueeze(0).to(device)

    # Start from pure noise
    x = torch.randn(1, ACTION_HORIZON * ACT_DIM, device=device)

    # Iteratively denoise
    for step in reversed(range(NUM_DIFFUSION_STEPS)):
        t = torch.tensor([[step / NUM_DIFFUSION_STEPS]], device=device)

        pred_noise = model(x, obs_t, t)

        alpha = alphas[step]
        alpha_cumprod = alphas_cumprod[step]
        beta = betas[step]

        # DDPM update
        if step > 0:
            noise = torch.randn_like(x)
        else:
            noise = torch.zeros_like(x)

        x = (1 / torch.sqrt(alpha)) * (
            x - (beta / torch.sqrt(1 - alpha_cumprod)) * pred_noise
        ) + torch.sqrt(beta) * noise

    actions = x.cpu().numpy()[0].reshape(ACTION_HORIZON, ACT_DIM)
    return np.clip(actions, -1, 1)


# ============================================================
# Evaluate
# ============================================================
print("\n" + "=" * 50)
print("Step 3: Evaluation — Diffusion Policy")
print("=" * 50)

env = gym.make("FetchReach-v4", render_mode=None)
successes = 0
num_eps = 50

for ep in range(num_eps):
    obs, _ = env.reset()
    action_queue = []

    for step in range(50):
        if len(action_queue) == 0:
            obs_vec = np.concatenate([obs["observation"], obs["desired_goal"]])
            chunk = sample_actions(obs_vec)
            action_queue = list(chunk)

        action = action_queue.pop(0)
        obs, _, term, trunc, info = env.step(action)

        if info.get("is_success", False):
            successes += 1
            break
        if term or trunc:
            break

env.close()

diff_sr = successes / num_eps * 100
print(f"\n  Diffusion Policy success rate: {diff_sr:.0f}%")
print(f"\n  Key insight: Diffusion generates actions by iterative denoising")
print(f"  {NUM_DIFFUSION_STEPS} denoising steps: noise -> clean action chunk")
print(f"  This handles multimodal distributions (multiple valid solutions)")

torch.save(model.state_dict(), "/Users/chan/robot-learning/diffusion_fetchreach.pt")
print(f"\nModel saved: diffusion_fetchreach.pt")
