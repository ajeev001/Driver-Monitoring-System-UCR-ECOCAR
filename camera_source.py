import cv2
import numpy as np

class CameraSource:
    def __init__(self, prefer_realsense=True, webcam_index=0, scan_indices=(0, 1, 2, 3), width=1280, height=720, fps=30):
        self.mode = None
        self.cap = None
        self.pipeline = None

        if prefer_realsense:
            try:
                import pyrealsense2 as rs
                self.rs = rs
                self.pipeline = rs.pipeline()
                config = rs.config()
                config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
                self.pipeline.start(config)
                self.mode = "realsense"
                print("[INFO] Camera: Intel RealSense D555")
                return
            except Exception:
                print("[WARN] RealSense not available, falling back to webcam")

        # Try the requested webcam index first, then fall back to scanning.
        tried_indices = []
        indices = [webcam_index] + [i for i in scan_indices if i != webcam_index]
        for idx in indices:
            tried_indices.append(idx)
            cap = cv2.VideoCapture(idx)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                cap.set(cv2.CAP_PROP_FPS, fps)
                self.cap = cap
                self.mode = "webcam"
                print(f"[INFO] Camera: Webcam index {idx}")
                break
            cap.release()

        if self.cap is None:
            raise RuntimeError(f"No usable camera found (tried indices: {tried_indices})")

        # self.mode and print handled above when a webcam opens

    def read(self):
        if self.mode == "realsense":
            frames = self.pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                return False, None
            frame = np.asanyarray(color_frame.get_data())
            return True, frame
        else:
            return self.cap.read()

    def release(self):
        if self.mode == "realsense":
            self.pipeline.stop()
        else:
            self.cap.release()
