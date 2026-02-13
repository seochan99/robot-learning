"""
Step 13: SmolVLA Inference — Image + Language -> Robot Actions

Actually run the VLA model:
1. Create a synthetic camera image
2. Give it a language command
3. Get robot actions out

This is the full VLA pipeline running on your MacBook!
"""

import torch
import numpy as np
import time

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Device: {device}")

# ============================================================
# Load SmolVLA + preprocessor
# ============================================================
print("\nLoading SmolVLA (450M params)...")
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.policies.factory import make_pre_post_processors

model_id = "lerobot/smolvla_base"
policy = SmolVLAPolicy.from_pretrained(model_id)
policy = policy.to(device)
policy.eval()

preprocess, postprocess = make_pre_post_processors(
    policy.config,
    model_id,
    preprocessor_overrides={"device_processor": {"device": str(device)}},
)

print(f"Model loaded! ({sum(p.numel() for p in policy.parameters()):,} params)")

# ============================================================
# Prepare raw inputs
# ============================================================
print("\n" + "=" * 55)
print("  VLA Input/Output Structure")
print("=" * 55)
print()
print("  Inputs:")
for key in policy.config.input_features:
    feat = policy.config.input_features[key]
    print(f"    {key}: {feat}")
print()
print("  Outputs:")
for key in policy.config.output_features:
    feat = policy.config.output_features[key]
    print(f"    {key}: {feat}")

# Create synthetic camera images (256x256 RGB)
np.random.seed(42)

# Build raw batch (before preprocessing)
raw_batch = {
    "observation.images.camera1": torch.randint(0, 255, (1, 3, 256, 256), dtype=torch.uint8),
    "observation.images.camera2": torch.randint(0, 255, (1, 3, 256, 256), dtype=torch.uint8),
    "observation.images.camera3": torch.randint(0, 255, (1, 3, 256, 256), dtype=torch.uint8),
    "observation.state": torch.FloatTensor([[0.1, -0.3, 0.5, 0.0, 0.2, -0.1]]),
    "task": ["pick up the red block"],
}

# Run through preprocessor (handles image normalization, tokenization, etc.)
print("\n" + "=" * 55)
print("  Preprocessing: raw data -> model-ready tensors")
print("=" * 55)

batch = preprocess(raw_batch)

print("  Preprocessed batch keys:")
for key in sorted(batch.keys()):
    if isinstance(batch[key], torch.Tensor):
        print(f"    {key}: {batch[key].shape} ({batch[key].dtype})")
    else:
        print(f"    {key}: {type(batch[key])}")

# ============================================================
# Run inference!
# ============================================================
print("\n" + "=" * 55)
print("  Running VLA inference: 'pick up the red block'")
print("=" * 55)

policy.reset()

t0 = time.time()
with torch.no_grad():
    action = policy.select_action(batch)
inference_time = time.time() - t0

action_np = action.cpu().numpy().flatten()

print(f"\n  Output actions (6 DOF):")
joint_names = ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "gripper"]
for name, val in zip(joint_names, action_np):
    bar = "#" * int(abs(val) * 20)
    direction = "+" if val >= 0 else "-"
    print(f"    {name:>10}: {val:>7.4f} [{direction}{bar}]")

print(f"\n  Inference time: {inference_time:.3f}s ({1/max(inference_time, 0.001):.1f} Hz)")

# ============================================================
# Try different language commands
# ============================================================
print("\n" + "=" * 55)
print("  Same image + different commands -> different actions")
print("=" * 55)

commands = [
    "pick up the red block",
    "push the cube to the left",
    "place the object on the plate",
    "open the gripper",
]

for cmd in commands:
    raw_batch["task"] = [cmd]
    batch = preprocess(raw_batch)
    policy.reset()
    with torch.no_grad():
        action = policy.select_action(batch)
    a = action.cpu().numpy().flatten()
    print(f"\n  '{cmd}'")
    print(f"    -> [{', '.join(f'{v:.3f}' for v in a)}]")

# ============================================================
# Summary
# ============================================================
print("\n\n" + "=" * 55)
print("  WHAT JUST HAPPENED")
print("=" * 55)
print(f"""
  You ran SmolVLA (450M params) on your MacBook M4 Pro!

  Pipeline:
    3x camera images (256x256)
    + robot state (6 joints)       ->  SmolVLA  ->  6 DOF actions
    + "pick up the red block"

  This is the SAME architecture used in real robots.
  The only difference: we used random images instead of real cameras.

  With real robot setup (xArm + cameras):
    1. Fine-tune SmolVLA on your teleoperation data
    2. "make an americano" -> SmolVLA -> xArm moves -> coffee!

  SmolVLA is a foundation model:
    - Pretrained on 10M+ frames from 487 datasets
    - Understands visual scenes + language commands
    - Fine-tune with YOUR data for YOUR task
""")
