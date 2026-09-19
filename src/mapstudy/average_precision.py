"""A from-scratch implementation of (mean) Average Precision.

It exists to make every step of the metric explicit, and to expose the choice of
precision interpolation, which the reference tools hard-code. With
``interpolation="coco"`` the results match ``pycocotools`` exactly (area range
"all", 100 detections per image and category); this is checked by the test suite.

Matching follows the COCO protocol: within each image, detections are processed in
decreasing score order and each one is greedily matched to the unmatched ground
truth with the highest IoU at or above the threshold. A detection that matches
nothing, including a duplicate of an already matched object, is a false positive.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np

from mapstudy.boxes import iou_matrix
from mapstudy.data import Detection, GroundTruth

Interpolation = Literal["coco", "voc", "none"]

COCO_IOU_THRESHOLDS = tuple(np.linspace(0.5, 0.95, 10))
COCO_RECALL_THRESHOLDS = np.linspace(0.0, 1.0, 101)
COCO_MAX_DETECTIONS = 100


@dataclass(frozen=True)
class PrecisionRecallCurve:
    """Cumulative precision and recall after each detection, sorted by decreasing score."""

    scores: np.ndarray
    precision: np.ndarray
    recall: np.ndarray
    n_ground_truths: int


def precision_recall_curve(
    detections: Sequence[Detection],
    ground_truths: Sequence[GroundTruth],
    iou_threshold: float,
    max_detections: int = COCO_MAX_DETECTIONS,
) -> PrecisionRecallCurve:
    """Build the precision-recall curve of a single category."""
    gts_by_image = _group_by_image(ground_truths)
    dts_by_image = _group_by_image(detections)

    scores: list[np.ndarray] = []
    is_tp: list[np.ndarray] = []
    for image_id in sorted(dts_by_image):
        image_scores, image_tp = _match_image(
            dts_by_image[image_id], gts_by_image.get(image_id, []), iou_threshold, max_detections
        )
        scores.append(image_scores)
        is_tp.append(image_tp)

    all_scores = np.concatenate(scores) if scores else np.empty(0)
    all_tp = np.concatenate(is_tp) if is_tp else np.empty(0, dtype=bool)
    order = np.argsort(-all_scores, kind="mergesort")
    all_scores, all_tp = all_scores[order], all_tp[order]

    tp_cumsum = np.cumsum(all_tp)
    fp_cumsum = np.cumsum(~all_tp)
    n_gts = len(ground_truths)
    return PrecisionRecallCurve(
        scores=all_scores,
        precision=tp_cumsum / np.maximum(tp_cumsum + fp_cumsum, 1),
        recall=tp_cumsum / n_gts if n_gts else np.zeros_like(tp_cumsum, dtype=float),
        n_ground_truths=n_gts,
    )


def average_precision(curve: PrecisionRecallCurve, interpolation: Interpolation = "coco") -> float:
    """Area under a precision-recall curve.

    - ``"coco"``: precision made monotonically decreasing, then sampled at 101 recall
      levels (MS-COCO).
    - ``"voc"``: precision made monotonically decreasing, then integrated exactly over
      every recall step (PASCAL VOC 2010 and later).
    - ``"none"``: raw precision integrated over every recall step. Shown for comparison
      only; it penalises the local dips that interpolation hides.
    """
    if curve.recall.size == 0:
        return 0.0

    precision = curve.precision
    if interpolation in ("coco", "voc"):
        precision = np.maximum.accumulate(precision[::-1])[::-1]

    if interpolation == "coco":
        indices = np.searchsorted(curve.recall, COCO_RECALL_THRESHOLDS, side="left")
        sampled = np.zeros_like(COCO_RECALL_THRESHOLDS)
        reachable = indices < precision.size
        sampled[reachable] = precision[indices[reachable]]
        return float(sampled.mean())

    if interpolation in ("voc", "none"):
        recall_steps = np.diff(curve.recall, prepend=0.0)
        return float(np.sum(recall_steps * precision))

    raise ValueError(f"Unknown interpolation: {interpolation!r}")


@dataclass(frozen=True)
class MeanAveragePrecision:
    value: float
    per_category: dict[int, float]


def mean_average_precision(
    ground_truths: Iterable[GroundTruth],
    detections: Iterable[Detection],
    iou_thresholds: Sequence[float] = (0.5,),
    interpolation: Interpolation = "coco",
    max_detections: int = COCO_MAX_DETECTIONS,
) -> MeanAveragePrecision:
    """AP averaged over IoU thresholds, then over categories that have ground truths.

    ``iou_thresholds=(0.5,)`` gives mAP@0.50; ``COCO_IOU_THRESHOLDS`` gives the COCO
    primary metric mAP@[0.50:0.95].
    """
    gts_by_category = _group_by_category(ground_truths)
    dts_by_category = _group_by_category(detections)

    per_category = {}
    for category_id, gts in sorted(gts_by_category.items()):
        dts = dts_by_category.get(category_id, [])
        aps = [
            average_precision(precision_recall_curve(dts, gts, threshold, max_detections), interpolation)
            for threshold in iou_thresholds
        ]
        per_category[category_id] = float(np.mean(aps))

    value = float(np.mean(list(per_category.values()))) if per_category else 0.0
    return MeanAveragePrecision(value, per_category)


@dataclass(frozen=True)
class OperatingPoint:
    """Counts and rates of a detector used at one fixed confidence threshold."""

    score_threshold: float
    true_positives: int
    false_positives: int
    false_negatives: int

    @property
    def precision(self) -> float:
        predicted = self.true_positives + self.false_positives
        return self.true_positives / predicted if predicted else 0.0

    @property
    def recall(self) -> float:
        actual = self.true_positives + self.false_negatives
        return self.true_positives / actual if actual else 0.0

    @property
    def f1(self) -> float:
        total = self.precision + self.recall
        return 2 * self.precision * self.recall / total if total else 0.0


def operating_point(
    ground_truths: Iterable[GroundTruth],
    detections: Iterable[Detection],
    score_threshold: float,
    iou_threshold: float = 0.5,
    max_detections: int = COCO_MAX_DETECTIONS,
) -> OperatingPoint:
    """Evaluate a detector as it would actually be deployed: at one confidence threshold.

    Unlike AP, which only sees the *ranking* of scores, this counts every detection kept
    by the threshold. It is what makes redundant predictions visible.
    """
    gts_by_category = _group_by_category(ground_truths)
    dts_by_category = _group_by_category(d for d in detections if d.score >= score_threshold)

    true_positives = false_positives = 0
    for category_id in gts_by_category.keys() | dts_by_category.keys():
        gts_by_image = _group_by_image(gts_by_category.get(category_id, []))
        for image_id, image_dts in _group_by_image(dts_by_category.get(category_id, [])).items():
            _, is_tp = _match_image(image_dts, gts_by_image.get(image_id, []), iou_threshold, max_detections)
            true_positives += int(is_tp.sum())
            false_positives += int((~is_tp).sum())

    n_gts = sum(len(gts) for gts in gts_by_category.values())
    return OperatingPoint(score_threshold, true_positives, false_positives, n_gts - true_positives)


def _match_image(
    detections: Sequence[Detection],
    ground_truths: Sequence[GroundTruth],
    iou_threshold: float,
    max_detections: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Greedy COCO matching within one image. Returns scores and true-positive flags."""
    order = np.argsort([-d.score for d in detections], kind="mergesort")[:max_detections]
    detections = [detections[i] for i in order]
    scores = np.array([d.score for d in detections], dtype=float)
    is_tp = np.zeros(len(detections), dtype=bool)
    if not ground_truths:
        return scores, is_tp

    ious = iou_matrix([d.box for d in detections], [g.box for g in ground_truths])
    matched = np.zeros(len(ground_truths), dtype=bool)
    for i in range(len(detections)):
        candidates = np.where(matched | (ious[i] < iou_threshold), -1.0, ious[i])
        best = int(np.argmax(candidates))
        if candidates[best] >= 0:
            matched[best] = True
            is_tp[i] = True
    return scores, is_tp


def _group_by_image(items: Iterable[Detection | GroundTruth]) -> dict[int, list]:
    groups: dict[int, list] = defaultdict(list)
    for item in items:
        groups[item.image_id].append(item)
    return groups


def _group_by_category(items: Iterable[Detection | GroundTruth]) -> dict[int, list]:
    groups: dict[int, list] = defaultdict(list)
    for item in items:
        groups[item.category_id].append(item)
    return groups
