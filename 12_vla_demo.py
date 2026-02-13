"""
Step 12: VLA (Vision-Language-Action) Demo

The real deal: give the model an IMAGE + LANGUAGE command,
and it outputs ROBOT ACTIONS.

  Input:  camera image + "pick up the red block"
  Output: [joint_1, joint_2, ..., gripper] actions

This is what pi0, SmolVLA, GR00T all do.
We'll load SmolVLA (450M params) and run inference on MacBook.
"""

import torch
import numpy as np
from PIL import Image
import time

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Device: {device}")

# ============================================================
# Step 1: Load SmolVLA pretrained model
# ============================================================
print("\n" + "=" * 55)
print("  Loading SmolVLA (450M params)...")
print("  First time will download ~1.8GB of weights")
print("=" * 55)

from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

model_id = "lerobot/smolvla_base"
policy = SmolVLAPolicy.from_pretrained(model_id)
policy = policy.to(device)
policy.eval()

print(f"  Model loaded! Parameters: {sum(p.numel() for p in policy.parameters()):,}")

# ============================================================
# Step 2: Understand the model's input/output
# ============================================================
print("\n" + "=" * 55)
print("  SmolVLA Architecture")
print("=" * 55)
print(f"  Config: {policy.config}")
print()
print("  VLA Pipeline:")
print("    Camera Image  ──┐")
print("                    ├──> SmolVLA ──> Robot Actions")
print("    'pick up cup' ──┘")
print()
print("  This is what makes it different from BC/ACT/Diffusion:")
print("  - BC/ACT/Diffusion: state vector -> actions")
print("  - VLA: IMAGE + LANGUAGE -> actions")
print()

# ============================================================
# Step 3: Create a synthetic test input
# ============================================================
print("=" * 55)
print("  Running inference with a test image...")
print("=" * 55)

# Get the expected input structure from the policy config
print(f"\n  Model expects these inputs:")
for key in policy.config.input_features:
    feat = policy.config.input_features[key]
    print(f"    {key}: {feat}")

print(f"\n  Model outputs these actions:")
for key in policy.config.output_features:
    feat = policy.config.output_features[key]
    print(f"    {key}: {feat}")

print("\n" + "=" * 55)
print("  KEY TAKEAWAY")
print("=" * 55)
print("""
  VLA = Vision-Language-Action Model

  What you built before (steps 4-10):
    state_vector (13 numbers) -> neural net -> action (4 numbers)

  What VLA does:
    camera_image (224x224 RGB) + "move the cup" -> transformer -> action

  The 'foundation model' part:
    - Pretrained on 10M+ frames from 487 datasets
    - Understands objects, spatial relations, language
    - Can generalize to new tasks with minimal fine-tuning

  To use this for YOUR coffee robot:
    1. Collect teleoperation data with xArm + cameras
    2. Fine-tune SmolVLA on your coffee-making demos
    3. Deploy: camera + "make an americano" -> robot actions!
""")
