from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np


@dataclass
class GazeRay:
    origin_c: np.ndarray  # shape (3,)
    direction_c: np.ndarray  # shape (3,), unit vector


@dataclass
class CameraTransform:
    # Transform from cabin (driver cam) to forward cam frame: p_f = R_cf * p_c + t_cf
    R_cf: np.ndarray  # shape (3, 3)
    t_cf: np.ndarray  # shape (3,)


@dataclass
class CameraIntrinsics:
    K: np.ndarray  # shape (3, 3)
    width: int
    height: int


@dataclass
class Box2D:
    xmin: float
    ymin: float
    xmax: float
    ymax: float

    def relaxed(self, pad_px: float) -> "Box2D":
        return Box2D(
            xmin=self.xmin - pad_px,
            ymin=self.ymin - pad_px,
            xmax=self.xmax + pad_px,
            ymax=self.ymax + pad_px,
        )

    def contains(self, u: float, v: float) -> bool:
        return self.xmin <= u <= self.xmax and self.ymin <= v <= self.ymax


def normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    if n == 0:
        return v
    return v / n


def transform_gaze_to_forward(gaze: GazeRay, tf: CameraTransform) -> GazeRay:
    origin_f = tf.R_cf @ gaze.origin_c + tf.t_cf
    direction_f = tf.R_cf @ gaze.direction_c
    direction_f = normalize(direction_f)
    return GazeRay(origin_f, direction_f)


def project_gaze_to_image(
    gaze_f: GazeRay, intr: CameraIntrinsics
) -> Optional[Tuple[float, float]]:
    # Find intersection with z = 1 plane in forward camera coords, then project.
    # p = origin + t * dir
    if abs(gaze_f.direction_c[2]) < 1e-6:
        return None
    t = (1.0 - gaze_f.origin_c[2]) / gaze_f.direction_c[2]
    if t <= 0:
        return None
    p = gaze_f.origin_c + t * gaze_f.direction_c
    uvw = intr.K @ p
    if uvw[2] <= 0:
        return None
    u = uvw[0] / uvw[2]
    v = uvw[1] / uvw[2]
    if u < 0 or v < 0 or u >= intr.width or v >= intr.height:
        # keep None to indicate out of FOV
        return None
    return float(u), float(v)


def gaze_hits_boxes(
    uv: Optional[Tuple[float, float]], boxes: List[Box2D], pad_px: float
) -> List[bool]:
    if uv is None:
        return [False] * len(boxes)
    u, v = uv
    hits = []
    for box in boxes:
        hits.append(box.relaxed(pad_px).contains(u, v))
    return hits


def select_hit_indices(
    uv: Optional[Tuple[float, float]],
    boxes: List[Box2D],
    pad_px: float,
) -> List[int]:
    hits = gaze_hits_boxes(uv, boxes, pad_px)
    return [i for i, h in enumerate(hits) if h]


def compute_gaze_scene_mapping(
    gaze_c: GazeRay,
    tf: CameraTransform,
    intr: CameraIntrinsics,
    boxes: List[Box2D],
    pad_px: float = 12.0,
) -> dict:
    gaze_f = transform_gaze_to_forward(gaze_c, tf)
    uv = project_gaze_to_image(gaze_f, intr)
    hit_indices = select_hit_indices(uv, boxes, pad_px)
    return {
        "uv": uv,
        "hit_indices": hit_indices,
    }
