import sys
import os
import time
import csv
from datetime import datetime

# =============================
# PATH SETUP
# =============================
ETHXGAZE_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

MODULE_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

sys.path.insert(0, ETHXGAZE_ROOT)
sys.path.insert(0, MODULE_ROOT)

import cv2
import numpy as np
import glob
from pathlib import Path

from camera.camera_source import CameraSource
from gaze.gaze_core import GazeEstimator


# =============================
# CONFIG
# =============================
GAZE_LENGTH_RATIO = 1 / 7
BASE_GAIN = 1.8
MAX_GAIN = 3.0

WEBCAM_INDEX = 0        # CHANGE if needed
LOG_CSV = True
CSV_DIR = "logs"
CALIBRATION_JSON = None  # If None, auto-load latest from logs/
CALIB_DIR = "/Users/jakobiparker/EcoCAR-CAV/Driver Monitoring System/ETH-XGaze/gaze_modules/logs"
SCREEN_WIDTH = 2560
SCREEN_HEIGHT = 1600


# =============================
# MAIN
# =============================
def main():

    # -----------------------------
    # Camera (force webcam, avoid phone)
    # -----------------------------
    camera = CameraSource(
        prefer_realsense=False,
        webcam_index=WEBCAM_INDEX
    )

    gaze = GazeEstimator(device="cpu")

    # -----------------------------
    # Optional calibration mapping
    # -----------------------------
    calib = None
    try:
        import json
        calib_path = CALIBRATION_JSON
        if calib_path is None:
            candidates = sorted(
                glob.glob(os.path.join(CALIB_DIR, "gaze_calibration_*.json")),
                key=os.path.getmtime,
            )
            if candidates:
                calib_path = candidates[-1]
        if calib_path:
            with open(calib_path, "r", encoding="utf-8") as f:
                calib = json.load(f)
            model_type = calib.get("model", {}).get("type", "affine")
            px = np.array(calib["model"]["params_x"], dtype=float)
            py = np.array(calib["model"]["params_y"], dtype=float)
            screen_w = int(calib["screen"]["width"])
            screen_h = int(calib["screen"]["height"])
            # Compute screen-space bias so the calibration center maps to screen center.
            screen_bias_x = 0.0
            screen_bias_y = 0.0
            points = calib.get("points", [])
            if points:
                cx = screen_w / 2.0
                cy = screen_h / 2.0
                center_pt = min(
                    points,
                    key=lambda p: (p.get("x", 0) - cx) ** 2 + (p.get("y", 0) - cy) ** 2,
                )
                cp = float(center_pt.get("pitch", 0.0))
                cyaw = float(center_pt.get("yaw", 0.0))
                a = np.array([cp, cyaw, 1.0], dtype=float)
                pred_cx = float(a @ px)
                pred_cy = float(a @ py)
                screen_bias_x = (screen_w / 2.0) - pred_cx
                screen_bias_y = (screen_h / 2.0) - pred_cy

            print(f"[INFO] Loaded calibration: {calib_path} (model={model_type})")
    except Exception as exc:
        print(f"[WARN] Failed to load calibration: {exc}")
        calib = None

    # -----------------------------
    # CSV logging setup
    # -----------------------------
    csv_file = None
    csv_writer = None
    frame_idx = 0

    if LOG_CSV:
        os.makedirs(CSV_DIR, exist_ok=True)
        csv_path = os.path.join(
            CSV_DIR,
            f"gaze_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        )

        csv_file = open(csv_path, "w", newline="")
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow([
            "timestamp",
            "frame_idx",
            "fps",
            "pitch_deg",
            "yaw_deg",
            "face_cx",
            "face_cy",
            "face_box_h",
            "gain",
            "gaze_x",
            "gaze_y",
            "screen_x",
            "screen_y",
            "camera_mode",
            "valid_detection"
        ])

        print(f"[INFO] CSV logging enabled: {csv_path}")

    prev_time = time.time()
    fps = 0.0
    smooth_alpha = 0.6
    smooth_pitch = None
    smooth_yaw = None
    endpoint_alpha = 0.7
    smooth_gx = None
    smooth_gy = None
    update_every = 1
    min_face_box_h = 120

    win_name = "Gaze Demo"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(win_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    # =============================
    # LOOP
    # =============================
    while True:
        ret, frame = camera.read()
        if not ret:
            break

        # -----------------------------
        # FPS calculation
        # -----------------------------
        now = time.time()
        fps = 0.9 * fps + 0.1 * (1.0 / max(1e-6, now - prev_time))
        prev_time = now

        result = gaze.process(frame)
        overlay_y = 25
        timestamp = time.time()

        if result:
            if frame_idx % update_every != 0:
                frame_idx += 1
                continue
            h, w = frame.shape[:2]

            cx, cy = result["face_center"]
            pitch, yaw = result["pitch"], result["yaw"]
            if smooth_pitch is None:
                smooth_pitch, smooth_yaw = pitch, yaw
            else:
                smooth_pitch = smooth_alpha * smooth_pitch + (1 - smooth_alpha) * pitch
                smooth_yaw = smooth_alpha * smooth_yaw + (1 - smooth_alpha) * yaw
            pitch, yaw = smooth_pitch, smooth_yaw
            box_h = result["face_box_h"]
            # Draw face bounding box if provided by the model
            face_box = result.get("face_box")
            if face_box is not None and len(face_box) == 4:
                x1, y1, x2, y2 = [int(v) for v in face_box]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            if box_h < min_face_box_h:
                frame_idx += 1
                continue

            length = int(min(h, w) * GAZE_LENGTH_RATIO)
            gain = BASE_GAIN * min(MAX_GAIN, 200 / box_h)

            dx = -gain * length * np.sin(yaw)
            dy = -gain * length * np.sin(pitch)

            gx, gy = int(cx + dx), int(cy + dy)

            # -----------------------------
            # Calibrated screen mapping
            # -----------------------------
            screen_pt = None
            img_gx, img_gy = gx, gy
            if calib is not None:
                a = np.array([pitch, yaw, 1.0], dtype=float)
                sx = float(a @ px) + screen_bias_x
                sy = float(a @ py) + screen_bias_y
                # clamp to screen bounds
                sx = max(0, min(screen_w - 1, sx))
                sy = max(0, min(screen_h - 1, sy))
                screen_pt = (int(sx), int(sy))
                # Map screen coords back to image coords using the same letterbox scaling.
                try:
                    scale = min(screen_w / w, screen_h / h)
                    new_w = int(w * scale)
                    new_h = int(h * scale)
                    x0 = (screen_w - new_w) // 2
                    y0 = (screen_h - new_h) // 2
                    ix = (sx - x0) / scale
                    iy = (sy - y0) / scale
                    ix = max(0, min(w - 1, ix))
                    iy = max(0, min(h - 1, iy))
                    img_gx, img_gy = int(ix), int(iy)
                except Exception:
                    img_gx, img_gy = gx, gy

            # -----------------------------
            # Draw gaze vector
            # -----------------------------
            if smooth_gx is None:
                smooth_gx, smooth_gy = img_gx, img_gy
            else:
                smooth_gx = endpoint_alpha * smooth_gx + (1 - endpoint_alpha) * img_gx
                smooth_gy = endpoint_alpha * smooth_gy + (1 - endpoint_alpha) * img_gy

            draw_gx, draw_gy = int(smooth_gx), int(smooth_gy)

            cv2.arrowedLine(
                frame,
                (int(cx), int(cy)),
                (draw_gx, draw_gy),
                (0, 255, 0),
                3,
                tipLength=0.25
            )
            cv2.circle(frame, (draw_gx, draw_gy), 4, (0, 0, 255), -1)

            # -----------------------------
            # Overlay metrics
            # -----------------------------
            metrics = [
                f"FPS: {fps:.1f}",
                f"Pitch: {np.degrees(pitch):.2f} deg",
                f"Yaw: {np.degrees(yaw):.2f} deg",
                f"Face box height: {box_h:.0f}px",
                f"Gain: {gain:.2f}",
                f"Gaze endpoint: ({draw_gx}, {draw_gy})",
                f"Camera: {camera.mode}"
            ]
            if screen_pt:
                metrics.insert(5, f"Screen gaze: ({screen_pt[0]}, {screen_pt[1]})")

            for line in metrics:
                cv2.putText(
                    frame,
                    line,
                    (15, overlay_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 255),
                    2
                )
                overlay_y += 22

            # -----------------------------
            # CSV logging (valid)
            # -----------------------------
            if csv_writer:
                csv_writer.writerow([
                    timestamp,
                    frame_idx,
                    round(fps, 2),
                    round(np.degrees(pitch), 3),
                    round(np.degrees(yaw), 3),
                    round(cx, 1),
                    round(cy, 1),
                    round(box_h, 1),
                    round(gain, 3),
                    gx,
                    gy,
                    "" if screen_pt is None else screen_pt[0],
                    "" if screen_pt is None else screen_pt[1],
                    camera.mode,
                    1
                ])

        else:
            cv2.putText(
                frame,
                f"FPS: {fps:.1f} | No face detected",
                (15, overlay_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2
            )

            # -----------------------------
            # CSV logging (invalid)
            # -----------------------------
            if csv_writer:
                csv_writer.writerow([
                    timestamp,
                    frame_idx,
                    round(fps, 2),
                    "", "", "", "", "", "", "", "",
                    "", "",
                    camera.mode,
                    0
                ])

        display_frame = frame
        target_w = screen_w if calib is not None else SCREEN_WIDTH
        target_h = screen_h if calib is not None else SCREEN_HEIGHT
        try:
            h, w = frame.shape[:2]
            scale = min(target_w / w, target_h / h)
            new_w = int(w * scale)
            new_h = int(h * scale)
            resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            display_frame = np.zeros((target_h, target_w, 3), dtype=resized.dtype)
            x0 = (target_w - new_w) // 2
            y0 = (target_h - new_h) // 2
            display_frame[y0:y0 + new_h, x0:x0 + new_w] = resized
        except Exception:
            display_frame = frame

        cv2.imshow(win_name, display_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        frame_idx += 1

    # =============================
    # CLEANUP
    # =============================
    camera.release()
    if csv_file:
        csv_file.close()
        print("[INFO] CSV file closed")

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
