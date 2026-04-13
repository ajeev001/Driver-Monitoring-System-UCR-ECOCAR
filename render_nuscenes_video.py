import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from nuscenes.nuscenes import NuScenes
from nuscenes.utils.geometry_utils import view_points
from nuscenes.utils.geometry_utils import BoxVisibility


def load_tracks(track_jsonl):
    tracks_by_frame = {}
    with open(track_jsonl, "r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            record = json.loads(line)
            tracks_by_frame[record["frame"]] = record["tracks"]
    return tracks_by_frame


def project_box(box, camera_intrinsic):
    corners = box.corners()  # 3x8 in sensor coords
    corners_2d = view_points(corners, camera_intrinsic, normalize=True)  # 3x8
    return corners_2d[:2].T  # 8x2


def draw_box(img, corners_2d, color=(0, 255, 0), thickness=2):
    # Draw the 3D box in 2D by connecting corners
    corners = corners_2d.astype(int)
    # 4 bottom corners: 0-3, 4 top corners: 4-7 (nuscenes order)
    # Connect bottom
    for i in range(4):
        p1 = tuple(corners[i])
        p2 = tuple(corners[(i + 1) % 4])
        cv2.line(img, p1, p2, color, thickness)
    # Connect top
    for i in range(4, 8):
        p1 = tuple(corners[i])
        p2 = tuple(corners[4 + (i + 1) % 4])
        cv2.line(img, p1, p2, color, thickness)
    # Connect verticals
    for i in range(4):
        p1 = tuple(corners[i])
        p2 = tuple(corners[i + 4])
        cv2.line(img, p1, p2, color, thickness)


def nearest_track_id(box_center, tracks, max_dist=5.0):
    best_id = None
    best_dist = None
    for trk in tracks:
        pos = np.array(trk["position"], dtype=float)
        dist = np.linalg.norm(pos - box_center)
        if best_dist is None or dist < best_dist:
            best_dist = dist
            best_id = trk["id"]
    if best_dist is None or best_dist > max_dist:
        return None
    return best_id


def render_video_from_scene(nusc, scene_name, sensor, tracks_by_frame, out_path, max_frames, fps):
    # Find scene
    scene = None
    for sc in nusc.scene:
        if sc["name"] == scene_name:
            scene = sc
            break
    if scene is None:
        raise ValueError(f"Scene not found: {scene_name}")

    # Prepare output writer using first frame size
    sample_token = scene["first_sample_token"]
    sample = nusc.get("sample", sample_token)
    sd_token = sample["data"][sensor]
    sample_data = nusc.get("sample_data", sd_token)
    img_path = Path(nusc.dataroot) / sample_data["filename"]
    img = cv2.imread(str(img_path))
    if img is None:
        raise FileNotFoundError(f"Could not read image: {img_path}")

    h, w = img.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))

    frame_idx = 0
    while sample_token:
        if max_frames is not None and frame_idx >= max_frames:
            break

        sample = nusc.get("sample", sample_token)
        sd_token = sample["data"][sensor]
        sample_data = nusc.get("sample_data", sd_token)
        img_path = Path(nusc.dataroot) / sample_data["filename"]
        img = cv2.imread(str(img_path))
        if img is None:
            sample_token = sample["next"]
            continue

        if "camera_intrinsic" in sample_data:
            cam_intrinsic = np.array(sample_data["camera_intrinsic"])
        else:
            calib = nusc.get("calibrated_sensor", sample_data["calibrated_sensor_token"])
            cam_intrinsic = np.array(calib["camera_intrinsic"])
        _, boxes, _ = nusc.get_sample_data(
            sd_token, box_vis_level=BoxVisibility.ANY, selected_anntokens=None
        )

        tracks = tracks_by_frame.get(frame_idx, [])
        for box in boxes:
            corners_2d = project_box(box, cam_intrinsic)
            draw_box(img, corners_2d, color=(0, 255, 0), thickness=2)
            track_id = nearest_track_id(box.center, tracks)
            if track_id is not None:
                center_2d = corners_2d.mean(axis=0).astype(int)
                cv2.putText(
                    img,
                    f"ID {track_id}",
                    (center_2d[0], center_2d[1]),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

        writer.write(img)
        frame_idx += 1
        sample_token = sample["next"]

    writer.release()


def render_video_from_tracks(nusc, track_records, out_path, max_frames, fps):
    if not track_records:
        raise ValueError("No track records provided.")

    first = track_records[0]
    if not first.get("filename"):
        raise ValueError("Track records missing filename.")

    img_path = Path(nusc.dataroot) / first["filename"]
    img = cv2.imread(str(img_path))
    if img is None:
        raise FileNotFoundError(f"Could not read image: {img_path}")

    h, w = img.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))

    frame_idx = 0
    for record in track_records:
        if max_frames is not None and frame_idx >= max_frames:
            break

        sd_token = record.get("sample_data_token")
        filename = record.get("filename")
        if not sd_token or not filename:
            continue

        img_path = Path(nusc.dataroot) / filename
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        sample_data = nusc.get("sample_data", sd_token)
        if "camera_intrinsic" in sample_data:
            cam_intrinsic = np.array(sample_data["camera_intrinsic"])
        else:
            calib = nusc.get("calibrated_sensor", sample_data["calibrated_sensor_token"])
            cam_intrinsic = np.array(calib["camera_intrinsic"])

        _, boxes, _ = nusc.get_sample_data(
            sd_token, box_vis_level=BoxVisibility.ANY, selected_anntokens=None
        )

        tracks = record.get("tracks", [])
        for box in boxes:
            corners_2d = project_box(box, cam_intrinsic)
            draw_box(img, corners_2d, color=(0, 255, 0), thickness=2)
            track_id = nearest_track_id(box.center, tracks)
            if track_id is not None:
                center_2d = corners_2d.mean(axis=0).astype(int)
                cv2.putText(
                    img,
                    f"ID {track_id}",
                    (center_2d[0], center_2d[1]),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

        writer.write(img)
        frame_idx += 1

    writer.release()


def main():
    parser = argparse.ArgumentParser(description="Render NuScenes video with AB3DMOT tracks.")
    parser.add_argument("--nusc-root", required=True)
    parser.add_argument("--version", default="v1.0-mini")
    parser.add_argument("--scene", default=None)
    parser.add_argument("--sensor", default="CAM_FRONT")
    parser.add_argument("--tracks", required=True, help="Path to JSONL track output.")
    parser.add_argument("--output", required=True, help="Output MP4 path.")
    parser.add_argument("--max-frames", type=int, default=50)
    parser.add_argument("--fps", type=int, default=60)
    args = parser.parse_args()

    nusc = NuScenes(version=args.version, dataroot=args.nusc_root, verbose=False)
    if args.scene:
        tracks_by_frame = load_tracks(args.tracks)
        render_video_from_scene(
            nusc,
            scene_name=args.scene,
            sensor=args.sensor,
            tracks_by_frame=tracks_by_frame,
            out_path=Path(args.output),
            max_frames=args.max_frames,
            fps=args.fps,
        )
    else:
        track_records = []
        with open(args.tracks, "r", encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                track_records.append(json.loads(line))
        render_video_from_tracks(
            nusc,
            track_records=track_records,
            out_path=Path(args.output),
            max_frames=args.max_frames,
            fps=args.fps,
        )


if __name__ == "__main__":
    main()
