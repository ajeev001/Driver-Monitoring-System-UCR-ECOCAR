import argparse
import json
import os
import sys
import time
from datetime import datetime

# =============================
# PATH SETUP (match other demos)
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

from camera.camera_source import CameraSource
from gaze.gaze_core import GazeEstimator

CALIB_DIR = "gaze_modules/logs"

def generate_grid_points(cols, rows, width, height, margin_ratio=0.1, include_center=True):
    mx = int(width * margin_ratio)
    my = int(height * margin_ratio)
    xs = np.linspace(mx, width - mx, cols).astype(int)
    ys = np.linspace(my, height - my, rows).astype(int)
    points = [(int(x), int(y)) for y in ys for x in xs]
    if include_center:
        cx, cy = int(width / 2), int(height / 2)
        if (cx, cy) not in points:
            points.append((cx, cy))
    return points


def build_linear_features(pitch_yaw):
    p = pitch_yaw[:, 0]
    y = pitch_yaw[:, 1]
    return np.column_stack([p, y, np.ones_like(p)])


def fit_linear_ridge(pitch_yaw, screen_xy, l2=1e-2):
    a = build_linear_features(pitch_yaw)
    x = screen_xy[:, 0]
    y = screen_xy[:, 1]
    ata = a.T @ a
    reg = l2 * np.eye(ata.shape[0])
    px = np.linalg.solve(ata + reg, a.T @ x)
    py = np.linalg.solve(ata + reg, a.T @ y)
    return px.tolist(), py.tolist()


def _detect_screen_size():
    try:
        import tkinter as tk

        root = tk.Tk()
        root.withdraw()
        w = int(root.winfo_screenwidth())
        h = int(root.winfo_screenheight())
        root.destroy()
        return w, h
    except Exception:
        return 1920, 1080


def main():
    parser = argparse.ArgumentParser(description="16-point gaze calibration (affine fit).")
    parser.add_argument("--cols", type=int, default=4, help="Grid columns.")
    parser.add_argument("--rows", type=int, default=4, help="Grid rows.")
    parser.add_argument("--frames-per-point", type=int, default=40, help="Samples per target.")
    parser.add_argument("--warmup-frames", type=int, default=10, help="Warmup frames per target.")
    parser.add_argument("--output", default=None, help="Output JSON path.")
    parser.add_argument(
        "--screen-width",
        type=int,
        default=0,
        help="Screen width in pixels (0 to auto-detect).",
    )
    parser.add_argument(
        "--screen-height",
        type=int,
        default=0,
        help="Screen height in pixels (0 to auto-detect).",
    )
    parser.add_argument("--webcam-index", type=int, default=0)
    args = parser.parse_args()

    screen_w, screen_h = args.screen_width, args.screen_height
    if screen_w <= 0 or screen_h <= 0:
        screen_w, screen_h = _detect_screen_size()
    points = generate_grid_points(args.cols, args.rows, screen_w, screen_h, include_center=True)

    camera = CameraSource(prefer_realsense=False, webcam_index=args.webcam_index)
    gaze = GazeEstimator(device="cpu")

    win_name = "Gaze Calibration"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(win_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.moveWindow(win_name, 0, 0)
    cv2.resizeWindow(win_name, screen_w, screen_h)

    collected = []
    frame_idx = 0

    for idx, (tx, ty) in enumerate(points):
        samples = []
        warmup = 0

        while len(samples) < args.frames_per_point:
            ret, frame = camera.read()
            if not ret:
                break

            canvas = np.zeros((screen_h, screen_w, 3), dtype=np.uint8)
            cv2.circle(canvas, (tx, ty), 10, (0, 255, 0), -1)
            cv2.putText(
                canvas,
                f"Point {idx + 1}/{len(points)}",
                (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (255, 255, 255),
                2,
            )
            cv2.putText(
                canvas,
                f"Samples {len(samples)}/{args.frames_per_point}",
                (30, 95),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (255, 255, 255),
                2,
            )

            result = gaze.process(frame)
            if result:
                if warmup >= args.warmup_frames:
                    samples.append([float(result["pitch"]), float(result["yaw"])])
                else:
                    warmup += 1

            cv2.imshow(win_name, canvas)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                camera.release()
                cv2.destroyAllWindows()
                return
            if key == ord("s"):
                break

            frame_idx += 1

        if samples:
            arr = np.array(samples, dtype=float)
            avg_pitch, avg_yaw = arr.mean(axis=0).tolist()
            collected.append(
                {
                    "x": int(tx),
                    "y": int(ty),
                    "pitch": avg_pitch,
                    "yaw": avg_yaw,
                    "samples": samples,
                }
            )

    camera.release()
    cv2.destroyAllWindows()

    if not collected:
        raise SystemExit("No samples collected.")

    pitch_yaw = np.array([[c["pitch"], c["yaw"]] for c in collected], dtype=float)
    screen_xy = np.array([[c["x"], c["y"]] for c in collected], dtype=float)
    px, py = fit_linear_ridge(pitch_yaw, screen_xy)

    out = {
        "screen": {"width": screen_w, "height": screen_h},
        "grid": {"cols": args.cols, "rows": args.rows},
        "frames_per_point": args.frames_per_point,
        "warmup_frames": args.warmup_frames,
        "points": collected,
        "model": {"type": "linear_ridge", "params_x": px, "params_y": py},
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }

    if args.output is None:
        os.makedirs(CALIB_DIR, exist_ok=True)
        # Remove previous auto-generated calibrations
        for name in os.listdir(CALIB_DIR):
            if name.startswith("gaze_calibration_") and name.endswith(".json"):
                try:
                    os.remove(os.path.join(CALIB_DIR, name))
                except OSError:
                    pass
        args.output = os.path.join(
            CALIB_DIR, f"gaze_calibration_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print(f"[INFO] Saved calibration to {args.output}")


if __name__ == "__main__":
    main()
