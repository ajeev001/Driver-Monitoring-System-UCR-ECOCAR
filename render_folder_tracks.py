import argparse
import json
from pathlib import Path

import cv2


def draw_box(img, bbox, color=(0, 255, 0), thickness=2):
    x1, y1, x2, y2 = [int(v) for v in bbox]
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)


def main():
    parser = argparse.ArgumentParser(description="Render MP4 from folder-tracking JSONL.")
    parser.add_argument("--tracks", required=True, help="Path to JSONL with filename + tracks.")
    parser.add_argument("--output", required=True, help="Output MP4 path.")
    parser.add_argument("--fps", type=int, default=30, help="FPS for output video.")
    parser.add_argument("--max-frames", type=int, default=None, help="Optional max frames.")
    args = parser.parse_args()

    track_records = []
    with open(args.tracks, "r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            track_records.append(json.loads(line))

    if not track_records:
        raise ValueError("No track records provided.")

    first = track_records[0]
    img_path = Path(first["filename"])
    img = cv2.imread(str(img_path))
    if img is None:
        raise FileNotFoundError(f"Could not read image: {img_path}")

    h, w = img.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(args.output), fourcc, args.fps, (w, h))

    frame_idx = 0
    for record in track_records:
        if args.max_frames is not None and frame_idx >= args.max_frames:
            break

        img_path = Path(record["filename"])
        img = cv2.imread(str(img_path))
        if img is None:
            frame_idx += 1
            continue

        tracks = record.get("tracks", [])
        for trk in tracks:
            bbox = trk.get("bbox")
            if not bbox:
                continue
            draw_box(img, bbox, color=(0, 255, 0), thickness=2)
            track_id = trk.get("id")
            if track_id is not None:
                x1, y1, _, _ = [int(v) for v in bbox]
                cv2.putText(
                    img,
                    f"ID {track_id}",
                    (x1, max(0, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

        writer.write(img)
        frame_idx += 1

    writer.release()


if __name__ == "__main__":
    main()
