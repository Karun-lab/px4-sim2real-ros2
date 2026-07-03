#!/usr/bin/env python3
"""
tello_rl_inference.py
=====================

Run your trained RL ball-tracking policy on a DJI Tello EDU.

Pipeline:
    Tello Camera Stream
            ↓
      Laptop Inference
            ↓
    RL action [vx, yaw]
            ↓
     send_rc_control()

Controls used:
    - forward/backward
    - yaw

Requirements:
    pip install djitellopy torch torchvision opencv-python numpy

Run:
    python3 tello_rl_inference.py
"""
import sys
sys.path.insert(0, "/home/karun/venvs/rl/lib/python3.12/site-packages")
import os

sys.path.append(
    os.path.expanduser("~/ros2_jazzy/src")
)
from sim2real.ball_tracking_inference import (
    detect_yellow,
    load_policy,
    build_obs_history,
    run_policy,
)

import time
import cv2
import numpy as np
import torch
from djitellopy import Tello
import traceback


# ── CONFIG ────────────────────────────────────────────────────────────────────
CHECKPOINT   = "/home/karun/ros2_jazzy/src/sim2real/sim2real/best_agent.pt"
IMG_H        = 64
IMG_W        = 64
HISTORY_LEN  = 3
CONTROL_HZ   = 15

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Tello speed limits
MAX_FB_SPEED  = 50          # cm/s, searching / slow approach
MAX_FB_SPEED_DETECTED = 60  # cm/s, boost when ball is visible  ← NEW
MAX_YAW_SPEED = 50          # deg/s

TARGET_HEIGHT = 150         # cm
HEIGHT_KP     = 0.35
MAX_UD_SPEED  = 20

# Safety
NO_DETECTION_TIMEOUT = 2.0  # seconds before forward motion is cut

# Flip yaw if drone turns the wrong way. Set to -1 to mirror.
YAW_SIGN = 1               # ← FLIPPED

# Search spin speed (deg/s) used when ball is not visible.
# Positive = clockwise (matches YAW_SIGN convention), negative = counter-clockwise.
SEARCH_YAW_SPEED = 25

# ── LOAD POLICY ───────────────────────────────────────────────────────────────
print("Loading policy...")
model = load_policy(CHECKPOINT, DEVICE)
model = model.to(DEVICE)
model.eval()
print("Policy loaded")

# ── TELLO SETUP ───────────────────────────────────────────────────────────────
tello = Tello()
tello.connect()
battery = tello.get_battery()
print(f"Battery: {battery}%")
tello.streamon()
frame_reader = tello.get_frame_read()

print("Taking off...")
tello.takeoff()

# Poll until drone actually reaches hover height (takeoff 'ok' fires early)
print("Waiting for stable hover...")
for _ in range(50):          # up to 5 seconds
    time.sleep(0.1)
    h = tello.get_distance_tof()
    print(f"  ToF: {h} cm")
    if h is not None and h > 50:
        break

time.sleep(1.0)              # brief settle
print(f"Hover confirmed at {tello.get_distance_tof()} cm — starting control loop")

# ── RL STATE ──────────────────────────────────────────────────────────────────
history             = None
last_detection_time = time.time()
prev_fb             = 0.0
prev_yaw            = 0.0
prev_ud             = 0.0

# ── MAIN LOOP ─────────────────────────────────────────────────────────────────
dt = 1.0 / CONTROL_HZ

try:
    while True:
        loop_start = time.time()

        # ── GET FRAME ────────────────────────────────────────────────────────
        frame = frame_reader.frame
        if frame is None:
            continue

        frame_rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_small = cv2.resize(frame_rgb, (IMG_W, IMG_H),
                                 interpolation=cv2.INTER_LINEAR)

        # ── BUILD OBSERVATION ────────────────────────────────────────────────
        rgb   = torch.tensor(frame_small, dtype=torch.float32,
                             device=DEVICE) / 255.0
        rgb_b = rgb.unsqueeze(0)          # (1, H, W, 3)

        # Yellow detection
        visible, yellow_mask, _ = detect_yellow(rgb_b, threshold=0.002)

        search_ch = (
            (~visible).float()
            .view(1, 1, 1, 1)
            .expand(1, IMG_H, IMG_W, 1)
        )

        frame_tensor = torch.cat([rgb_b, search_ch], dim=-1)

        history = build_obs_history(frame_tensor, history, HISTORY_LEN)

        # ── RUN RL POLICY (always, to keep history warm) ─────────────────────
        action = run_policy(model, history)
        vx  = float(action[0])
        yaw = float(action[1])

        ball_visible    = visible.item()
        time_since_seen = time.time() - last_detection_time

        if ball_visible:
            last_detection_time = time.time()
            time_since_seen     = 0.0

        # ── MODE SWITCH: RL tracking vs hardcoded search spin ────────────────
        if ball_visible:
            # ── RL IN CONTROL ────────────────────────────────────────────────
            alpha  = 0.7
            fb_speed  = alpha * prev_fb  + (1 - alpha) * (vx * MAX_FB_SPEED_DETECTED)
            yaw_speed = alpha * prev_yaw + (1 - alpha) * (yaw * MAX_YAW_SPEED * YAW_SIGN)

            prev_fb  = fb_speed
            prev_yaw = yaw_speed

            fb_speed  = int(np.clip(fb_speed,  -MAX_FB_SPEED_DETECTED, MAX_FB_SPEED_DETECTED))
            yaw_speed = int(np.clip(yaw_speed, -MAX_YAW_SPEED,         MAX_YAW_SPEED))

        else:
            # ── SEARCH SPIN: ignore RL, just rotate in place ─────────────────
            # Reset smoothing so there's no momentum carry-over when RL resumes
            prev_fb  = 0.0
            prev_yaw = 0.0

            fb_speed  = 0
            yaw_speed = SEARCH_YAW_SPEED    # constant slow spin

        # ── HEIGHT HOLD ──────────────────────────────────────────────────────
        try:
            current_height = tello.get_distance_tof()
            if current_height is None or current_height < 20:
                # Sensor reading bad — push up gently, don't zero out
                ud_speed = 15
            else:
                height_error = TARGET_HEIGHT - current_height
                ud_speed     = HEIGHT_KP * height_error
        except Exception:
            ud_speed = 15

        ud_speed = 0.7 * prev_ud + 0.3 * ud_speed
        prev_ud  = ud_speed
        ud_speed = int(np.clip(ud_speed, -MAX_UD_SPEED, MAX_UD_SPEED))

        # Store for display before current_height might not exist
        display_height = current_height if 'current_height' in dir() else TARGET_HEIGHT

        # ── SEND RC CONTROL ──────────────────────────────────────────────────
        # send_rc_control(left_right, forward_backward, up_down, yaw)
        tello.send_rc_control(0, fb_speed, ud_speed, yaw_speed)

        # ── DEBUG VIEW ───────────────────────────────────────────────────────
        # Main camera feed (640×480)
        debug_frame = cv2.resize(frame_rgb, (640, 480))
        # Note: frame_rgb is RGB, cv2.imshow expects BGR
        # debug_frame = cv2.cvtColor(debug_frame, cv2.COLOR_RGB2BGR)

        # ── Yellow mask inset (top-right corner) ─────────────────────────────
        # Build a yellow-highlighted version of the small frame
        # yellow_mask shape: (1, H, W) or (H, W) bool tensor — squeeze to numpy
        mask_np = yellow_mask.squeeze().cpu().numpy().astype(np.uint8)  # 0 or 1

        # Create a coloured overlay: yellow where detected, grey elsewhere
        mask_vis = np.zeros((IMG_H, IMG_W, 3), dtype=np.uint8)
        mask_vis[mask_np == 0] = [40, 40, 40]    # dark grey background
        mask_vis[mask_np == 1] = [0, 255, 255]   # cyan-yellow in BGR

        # Scale up to 160×160 for visibility and place top-right
        inset_size  = 160
        mask_inset  = cv2.resize(mask_vis, (inset_size, inset_size),
                                 interpolation=cv2.INTER_NEAREST)
        inset_x     = 640 - inset_size - 10
        inset_y     = 10
        debug_frame[inset_y:inset_y + inset_size,
                    inset_x:inset_x + inset_size] = mask_inset

        # Border around inset
        cv2.rectangle(debug_frame,
                      (inset_x - 1, inset_y - 1),
                      (inset_x + inset_size, inset_y + inset_size),
                      (200, 200, 200), 1)
        cv2.putText(debug_frame, "Yellow mask",
                    (inset_x, inset_y - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

        # ── HUD text ─────────────────────────────────────────────────────────
        status      = "RL TRACKING"   if ball_visible else "SEARCH SPIN"
        status_col  = (0, 200, 0)     if ball_visible else (0, 165, 255)  # green / orange BGR

        hud = [
            (f"vx: {vx:.2f}",              (0, 200, 0)),
            (f"yaw: {yaw:.2f}",            (200, 100, 0)),
            (f"FB: {fb_speed}",            (0, 0, 200)),
            (f"Yaw cmd: {yaw_speed}",      (200, 200, 0)),
            (status,                        status_col),
            (f"Height: {display_height} cm",(200, 200, 200)),
            (f"UD: {ud_speed}",            (200, 0, 200)),
            (f"No-det: {time_since_seen:.1f}s", (100, 100, 255)),
        ]

        for i, (txt, col) in enumerate(hud):
            cv2.putText(debug_frame, txt,
                        (20, 40 + i * 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.85, col, 2)

        cv2.imshow("Tello RL Tracking", debug_frame)

        # ── EXIT on 'q' ──────────────────────────────────────────────────────
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # ── RATE LIMIT ───────────────────────────────────────────────────────
        elapsed    = time.time() - loop_start
        sleep_time = dt - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)

except Exception as e:
    print("ERROR:", e)
    traceback.print_exc()

finally:
    print("Landing...")
    tello.send_rc_control(0, 0, 0, 0)
    time.sleep(1)
    tello.land()
    tello.streamoff()
    cv2.destroyAllWindows()
    print("Done")