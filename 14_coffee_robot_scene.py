"""
Step 14: Coffee Robot Scene — Custom MuJoCo Environment

A good-looking robot arm scene with:
- 6DOF robot arm with gripper
- Coffee machine, cups, saucer
- Realistic materials and lighting
- Interactive: control with keyboard

Controls (in terminal):
  W/S  — joint 2 (shoulder)
  A/D  — joint 1 (base rotation)
  Q/E  — joint 3 (elbow)
  Z/C  — joint 5 (wrist)
  O/P  — gripper open/close
  1-6  — select joint to control with W/S
  R    — reset scene
  X    — quit
"""

import mujoco
import mujoco.viewer
import numpy as np
import sys
import tty
import termios
import select
import time
import os

# ============================================================
# Keyboard
# ============================================================
old_settings = termios.tcgetattr(sys.stdin)


def init_kb():
    tty.setcbreak(sys.stdin.fileno())


def restore_kb():
    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)


def get_key():
    if select.select([sys.stdin], [], [], 0)[0]:
        return sys.stdin.read(1).lower()
    return None


# ============================================================
# Load scene
# ============================================================
xml_path = os.path.join(os.path.dirname(__file__), "assets", "coffee_scene.xml")
model = mujoco.MjModel.from_xml_path(xml_path)
data = mujoco.MjData(model)

# Set initial robot pose (arm reaching forward)
data.qpos[0] = 0.0    # joint1 (base)
data.qpos[1] = -0.5   # joint2 (shoulder)
data.qpos[2] = 0.8    # joint3 (elbow)
data.qpos[3] = 0.0    # joint4
data.qpos[4] = 0.3    # joint5 (wrist)
data.qpos[5] = 0.0    # joint6

mujoco.mj_forward(model, data)

# ============================================================
# Control
# ============================================================
ctrl = np.zeros(model.nu)
active_joint = 1  # which joint W/S controls (0-5)
gripper_val = 0.0
SPEED = 0.3

print("=" * 55)
print("  COFFEE ROBOT SCENE")
print("=" * 55)
print("  A/D = base rotation  |  W/S = active joint")
print("  Q/E = elbow          |  Z/C = wrist")
print("  O/P = gripper        |  1-6 = select joint")
print("  R   = reset          |  X   = quit")
print("=" * 55)

init_kb()

try:
    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.azimuth = 135
        viewer.cam.elevation = -25
        viewer.cam.distance = 1.8
        viewer.cam.lookat[:] = [0.45, 0, 0.55]

        while viewer.is_running():
            key = get_key()

            if key == "x":
                break

            if key == "r":
                mujoco.mj_resetData(model, data)
                data.qpos[1] = -0.5
                data.qpos[2] = 0.8
                data.qpos[4] = 0.3
                ctrl[:] = 0
                print("\r  [RESET]        ")

            # Joint selection
            if key in ("1", "2", "3", "4", "5", "6"):
                active_joint = int(key) - 1
                print(f"\r  Active joint: {active_joint + 1}        ")

            # Base rotation
            if key == "a": ctrl[0] -= SPEED
            if key == "d": ctrl[0] += SPEED

            # Active joint
            if key == "w": ctrl[active_joint] += SPEED
            if key == "s": ctrl[active_joint] -= SPEED

            # Elbow
            if key == "q": ctrl[2] -= SPEED
            if key == "e": ctrl[2] += SPEED

            # Wrist
            if key == "z": ctrl[4] -= SPEED
            if key == "c": ctrl[4] += SPEED

            # Gripper
            if key == "o":
                gripper_val = max(0, gripper_val - 0.3)
                print(f"\r  Gripper: {'OPEN' if gripper_val < 0.5 else 'CLOSED'}        ")
            if key == "p":
                gripper_val = min(1, gripper_val + 0.3)
                print(f"\r  Gripper: {'OPEN' if gripper_val < 0.5 else 'CLOSED'}        ")

            ctrl[6] = gripper_val
            ctrl[7] = gripper_val

            # Damping: controls decay toward zero
            ctrl *= 0.95

            data.ctrl[:] = ctrl
            mujoco.mj_step(model, data)
            viewer.sync()

            time.sleep(0.01)

except Exception as e:
    print(f"\nError: {e}")
finally:
    restore_kb()
    print("Done!")
