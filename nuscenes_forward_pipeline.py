import argparse
import json
import sys
from pathlib import Path

import numpy as np

from ab3dmot import MultiObjectTracker


def _import_nuscenes():
    try:
        from nuscenes.nuscenes import NuScenes
        from nuscenes.utils.data_classes import Box
        from nuscenes.utils.geometry_utils import BoxVisibility
    except Exception as exc:
        raise RuntimeError(
            "NuScenes devkit is required. Install it with:\n"
            "  python3 -m pip install nuscenes-devkit"
        ) from exc
    return NuScenes, Box, BoxVisibility


def _box_yaw(box):
    # NuScenes Box uses a quaternion; yaw_pitch_roll returns yaw in radians.
    try:
        yaw, _, _ = box.orientation.yaw_pitch_roll
        return float(yaw)
    except Exception:
        return 0.0


def sample_to_detections(nusc, sample_data_token, box_visibility):
    """
    Convert a NuScenes sample into detection dicts for AB3DMOT.
    Uses ground-truth boxes as detections to bootstrap the pipeline.
    """
    _, boxes, _ = nusc.get_sample_data(
        sample_data_token,
        box_vis_level=box_visibility,
        selected_anntokens=None,
    )

    detections = []
    for box in boxes:
        label = box.name
        pos = [float(box.center[0]), float(box.center[1]), float(box.center[2])]
        dims = [float(box.wlh[0]), float(box.wlh[1]), float(box.wlh[2])]
        heading = _box_yaw(box)
        detections.append(
            {
                "position": pos,
                "dimensions": dims,
                "score": 1.0,
                "label": label,
                "heading": heading,
            }
        )
    return detections


def iter_scene_samples(nusc, scene_name):
    for scene in nusc.scene:
        if scene["name"] == scene_name:
            token = scene["first_sample_token"]
            while token:
                sample = nusc.get("sample", token)
                yield sample
                token = sample["next"]
            return
    raise ValueError(f"Scene not found: {scene_name}")


def run_sequence(
    nusc,
    scene_name,
    sensor_channel,
    box_visibility,
    max_frames=None,
    output_path=None,
    sample_data_tokens=None,
):
    tracker = MultiObjectTracker(
        {
            "dt": 0.1,
            "initial_covariance": 100,
            "cost_threshold": 4.0,
            "confirmation_frames_needed": 3,
            "confirmation_window": 5,
            "deletion_missed_threshold": 3,
            "deletion_window": 5,
        }
    )

    out_fh = open(output_path, "w", encoding="utf-8") if output_path else None

    prev_ts = None
    frame_idx = 0

    try:
        if sample_data_tokens is None:
            iterable = iter_scene_samples(nusc, scene_name)
            for sample in iterable:
                sensor_token = sample["data"].get(sensor_channel)
                if sensor_token is None:
                    raise ValueError(f"Sensor channel not found in sample: {sensor_channel}")

                sample_data = nusc.get("sample_data", sensor_token)
                ts = sample_data["timestamp"]  # microseconds
                if prev_ts is None:
                    dt = 0.1
                else:
                    dt = max((ts - prev_ts) / 1e6, 1e-3)
                prev_ts = ts

                detections = sample_to_detections(nusc, sensor_token, box_visibility)

                tracker.predict(dt)
                tracks = tracker.update(detections, dt)

                if out_fh:
                    out_fh.write(
                        json.dumps(
                            {
                                "frame": frame_idx,
                                "timestamp_us": ts,
                                "sample_data_token": sensor_token,
                                "filename": sample_data.get("filename"),
                                "tracks": tracks,
                                "detections": detections,
                            }
                        )
                        + "\n"
                    )
                else:
                    print(f"frame={frame_idx} tracks={len(tracks)} detections={len(detections)}")

                frame_idx += 1
                if max_frames is not None and frame_idx >= max_frames:
                    break
        else:
            for sensor_token in sample_data_tokens:
                sample_data = nusc.get("sample_data", sensor_token)
                ts = sample_data["timestamp"]  # microseconds
                if prev_ts is None:
                    dt = 0.1
                else:
                    dt = max((ts - prev_ts) / 1e6, 1e-3)
                prev_ts = ts

                detections = sample_to_detections(nusc, sensor_token, box_visibility)

                tracker.predict(dt)
                tracks = tracker.update(detections, dt)

                if out_fh:
                    out_fh.write(
                        json.dumps(
                            {
                                "frame": frame_idx,
                                "timestamp_us": ts,
                                "sample_data_token": sensor_token,
                                "filename": sample_data.get("filename"),
                                "tracks": tracks,
                                "detections": detections,
                            }
                        )
                        + "\n"
                    )
                else:
                    print(f"frame={frame_idx} tracks={len(tracks)} detections={len(detections)}")

                frame_idx += 1
                if max_frames is not None and frame_idx >= max_frames:
                    break
    finally:
        if out_fh:
            out_fh.close()


def _image_detections_yolo(model, image_path, conf, imgsz):
    result = model.predict(source=str(image_path), conf=conf, imgsz=imgsz, verbose=False)[0]
    detections = []
    if result.boxes is None:
        return detections
    boxes = result.boxes
    xyxy = boxes.xyxy.cpu().numpy()
    scores = boxes.conf.cpu().numpy()
    classes = boxes.cls.cpu().numpy().astype(int)
    names = getattr(model, "names", {})
    for (x1, y1, x2, y2), score, cls in zip(xyxy, scores, classes):
        cx = float((x1 + x2) / 2.0)
        cy = float((y1 + y2) / 2.0)
        w = float(x2 - x1)
        h = float(y2 - y1)
        label = names.get(cls, cls)
        detections.append(
            {
                "position": [cx, cy, 0.0],
                "dimensions": [w, h, 0.0],
                "score": float(score),
                "label": label,
                "heading": 0.0,
                "bbox": [float(x1), float(y1), float(x2), float(y2)],
            }
        )
    return detections


def run_image_folder(
    image_dir,
    fps,
    output_path=None,
    yolo_weights="yolov8n.pt",
    yolo_conf=0.25,
    yolo_imgsz=640,
):
    try:
        from ultralytics import YOLO
    except Exception as exc:
        raise RuntimeError(
            "Ultralytics is required for folder mode. Install it with:\n"
            "  python3 -m pip install ultralytics"
        ) from exc

    image_dir = Path(image_dir)
    if not image_dir.exists():
        raise FileNotFoundError(f"Image dir not found: {image_dir}")

    image_files = sorted(
        [
            p
            for p in image_dir.iterdir()
            if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}
        ]
    )
    if not image_files:
        raise ValueError(f"No image files found in: {image_dir}")

    model = YOLO(yolo_weights)
    tracker = MultiObjectTracker(
        {
            "dt": 1.0 / fps,
            "initial_covariance": 100,
            "cost_threshold": 50.0,
            "confirmation_frames_needed": 3,
            "confirmation_window": 5,
            "deletion_missed_threshold": 3,
            "deletion_window": 5,
        }
    )

    out_fh = open(output_path, "w", encoding="utf-8") if output_path else None
    try:
        for frame_idx, image_path in enumerate(image_files):
            detections = _image_detections_yolo(
                model, image_path, conf=yolo_conf, imgsz=yolo_imgsz
            )
            tracker.predict(1.0 / fps)
            tracks = tracker.update(detections, 1.0 / fps)

            # Add 2D bbox to tracks for rendering
            tracks_out = []
            for trk in tracks:
                cx, cy, _ = trk["position"]
                w, h, _ = trk["dimensions"]
                x1 = float(cx - w / 2.0)
                y1 = float(cy - h / 2.0)
                x2 = float(cx + w / 2.0)
                y2 = float(cy + h / 2.0)
                trk_out = dict(trk)
                trk_out["bbox"] = [x1, y1, x2, y2]
                tracks_out.append(trk_out)

            if out_fh:
                out_fh.write(
                    json.dumps(
                        {
                            "frame": frame_idx,
                            "timestamp_us": int(frame_idx * (1.0 / fps) * 1e6),
                            "filename": str(image_path),
                            "tracks": tracks_out,
                            "detections": detections,
                        }
                    )
                    + "\n"
                )
            else:
                print(
                    f"frame={frame_idx} tracks={len(tracks_out)} detections={len(detections)}"
                )
    finally:
        if out_fh:
            out_fh.close()


def collect_existing_sample_data_tokens(nusc, sensor_channel, max_frames=None):
    tokens = []
    for sd in nusc.sample_data:
        if sd.get("channel") != sensor_channel:
            continue
        filename = sd.get("filename")
        if not filename:
            continue
        img_path = Path(nusc.dataroot) / filename
        if img_path.exists():
            tokens.append((sd["timestamp"], sd["token"]))
    tokens.sort(key=lambda x: x[0])
    ordered = [t[1] for t in tokens]
    if max_frames is not None:
        ordered = ordered[:max_frames]
    return ordered


def main():
    parser = argparse.ArgumentParser(description="NuScenes forward perception pipeline using AB3DMOT.")
    parser.add_argument("--nusc-root", default=None, help="Path to NuScenes dataset root.")
    parser.add_argument("--version", default="v1.0-mini", help="NuScenes version, e.g., v1.0-mini, v1.0-trainval.")
    parser.add_argument("--scene", default=None, help="Scene name, e.g., scene-0061.")
    parser.add_argument("--sensor", default="LIDAR_TOP", help="Sensor channel, e.g., LIDAR_TOP or CAM_FRONT.")
    parser.add_argument("--max-frames", type=int, default=None, help="Max frames to process.")
    parser.add_argument("--output", default=None, help="Optional JSONL output path.")
    parser.add_argument("--image-dir", default=None, help="Process a folder of images instead of NuScenes scenes.")
    parser.add_argument("--fps", type=float, default=10.0, help="FPS for folder mode (used for dt).")
    parser.add_argument("--yolo-weights", default="yolov8n.pt", help="YOLOv8 weights path for folder mode.")
    parser.add_argument("--yolo-conf", type=float, default=0.25, help="YOLOv8 confidence threshold.")
    parser.add_argument("--yolo-imgsz", type=int, default=640, help="YOLOv8 inference image size.")
    parser.add_argument(
        "--use-existing-files",
        action="store_true",
        help="Use all sample_data entries whose files exist, ordered by timestamp (ignores --scene).",
    )
    args = parser.parse_args()

    if args.image_dir:
        run_image_folder(
            image_dir=args.image_dir,
            fps=args.fps,
            output_path=args.output,
            yolo_weights=args.yolo_weights,
            yolo_conf=args.yolo_conf,
            yolo_imgsz=args.yolo_imgsz,
        )
        return

    if not args.nusc_root:
        raise ValueError("--nusc-root is required unless --image-dir is set.")

    NuScenes, _, BoxVisibility = _import_nuscenes()
    nusc = NuScenes(version=args.version, dataroot=args.nusc_root, verbose=True)

    if args.use_existing_files:
        tokens = collect_existing_sample_data_tokens(
            nusc, sensor_channel=args.sensor, max_frames=args.max_frames
        )
        run_sequence(
            nusc,
            scene_name=None,
            sensor_channel=args.sensor,
            box_visibility=BoxVisibility.ANY,
            max_frames=args.max_frames,
            output_path=args.output,
            sample_data_tokens=tokens,
        )
    else:
        if not args.scene:
            raise ValueError("--scene is required unless --use-existing-files is set.")
        run_sequence(
            nusc,
            scene_name=args.scene,
            sensor_channel=args.sensor,
            box_visibility=BoxVisibility.ANY,
            max_frames=args.max_frames,
            output_path=args.output,
        )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
