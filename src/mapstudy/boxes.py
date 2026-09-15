"""Bounding-box geometry.

All boxes follow the COCO convention ``[x, y, width, height]`` where ``(x, y)`` is
the top-left corner, in pixels.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

Box = Sequence[float]


def box_area(box: Box) -> float:
    """Area of a box; degenerate boxes (non-positive width or height) have area 0."""
    return max(0.0, box[2]) * max(0.0, box[3])


def intersection_area(a: Box, b: Box) -> float:
    """Area of the overlap between two boxes."""
    left = max(a[0], b[0])
    top = max(a[1], b[1])
    right = min(a[0] + a[2], b[0] + b[2])
    bottom = min(a[1] + a[3], b[1] + b[3])
    return max(0.0, right - left) * max(0.0, bottom - top)


def iou(a: Box, b: Box) -> float:
    """Intersection over Union of two boxes, in ``[0, 1]``."""
    inter = intersection_area(a, b)
    union = box_area(a) + box_area(b) - inter
    return inter / union if union > 0 else 0.0


def iou_matrix(boxes_a: Sequence[Box], boxes_b: Sequence[Box]) -> np.ndarray:
    """Pairwise IoU, shape ``(len(boxes_a), len(boxes_b))``."""
    a = np.asarray(boxes_a, dtype=float).reshape(-1, 4)
    b = np.asarray(boxes_b, dtype=float).reshape(-1, 4)

    left = np.maximum(a[:, None, 0], b[None, :, 0])
    top = np.maximum(a[:, None, 1], b[None, :, 1])
    right = np.minimum(a[:, None, 0] + a[:, None, 2], b[None, :, 0] + b[None, :, 2])
    bottom = np.minimum(a[:, None, 1] + a[:, None, 3], b[None, :, 1] + b[None, :, 3])
    inter = np.clip(right - left, 0, None) * np.clip(bottom - top, 0, None)

    area_a = np.clip(a[:, 2], 0, None) * np.clip(a[:, 3], 0, None)
    area_b = np.clip(b[:, 2], 0, None) * np.clip(b[:, 3], 0, None)
    union = area_a[:, None] + area_b[None, :] - inter

    return np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)
