"""Two properties of AP that the leaderboard number hides.

Both are statements about what AP *cannot* see, and both are checked here rather than
argued:

- :func:`score_invariance` — AP depends on the ranking of confidence scores and on
  nothing else. Replacing every score by any strictly increasing function of it leaves
  AP bit-for-bit identical, so AP carries no information about the values themselves and
  cannot say at what confidence a detector should be deployed.
- :func:`interpolation_bias` — the monotone envelope applied before integrating the
  precision-recall curve only ever raises AP, and the amount it raises it by grows as
  the number of objects falls. The metric is therefore most generous exactly where
  evidence is thinnest: on rare categories.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass

import numpy as np

from mapstudy.average_precision import (
    _group_by_category,
    average_precision,
    mean_average_precision,
    operating_point,
    precision_recall_curve,
)
from mapstudy.data import Detection, ImageRecord
from mapstudy.evaluation import evaluate_lrp
from mapstudy.scenarios import generate_detections

logger = logging.getLogger(__name__)

DEFAULT_SCORE_THRESHOLD = 0.05

#: Strictly increasing maps of ``(0, 1)`` onto itself. Applying any of them to every
#: confidence score preserves the ranking of the detections and changes nothing else.
#: ``squashed`` is the interesting one: it compresses every score into ``[0.10, 0.12]``,
#: which destroys any possibility of choosing a threshold while leaving AP untouched.
MONOTONE_MAPS: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "identity": lambda s: s,
    "cubed": lambda s: s**3,
    "square root": lambda s: np.sqrt(s),
    "logistic": lambda s: 1.0 / (1.0 + np.exp(-12.0 * (s - 0.5))),
    "squashed": lambda s: 0.10 + 0.02 * s,
}


@dataclass(frozen=True)
class InvariancePoint:
    """One monotone remapping of the confidence scores, seen by every metric."""

    transform: str
    score_min: float
    score_max: float
    map50: float
    map: float
    olrp: float | None
    olrp_optimal_threshold: float | None
    precision: float
    recall: float
    f1: float

    def to_dict(self) -> dict:
        return asdict(self)


def remap_scores(
    detections: Sequence[Detection], transform: Callable[[np.ndarray], np.ndarray]
) -> list[Detection]:
    """Apply a monotone map to every confidence score, leaving boxes and labels alone."""
    scores = transform(np.array([d.score for d in detections], dtype=float))
    return [
        Detection(d.image_id, d.category_id, d.box, float(score))
        for d, score in zip(detections, scores, strict=True)
    ]


def score_invariance(
    images: Sequence[ImageRecord],
    *,
    scenario: str = "baseline",
    seed: int,
    score_threshold: float = DEFAULT_SCORE_THRESHOLD,
    transforms: dict[str, Callable[[np.ndarray], np.ndarray]] | None = None,
) -> list[InvariancePoint]:
    """Evaluate one detector under several monotone remappings of its scores.

    The detections, and therefore the boxes, are identical in every row: only the
    numbers attached to them change, and they change in a way that preserves their
    order. Any metric that reads the ranking alone must return the same value in every
    row; any metric that reads the values themselves must not.
    """
    ground_truths = [gt for image in images for gt in image.objects]
    detections = generate_detections(images, scenario, seed)

    points = []
    for name, transform in (transforms or MONOTONE_MAPS).items():
        remapped = remap_scores(detections, transform)
        scores = [d.score for d in remapped]
        lrp = evaluate_lrp(images, remapped)
        point = operating_point(ground_truths, remapped, score_threshold)
        points.append(
            InvariancePoint(
                transform=name,
                score_min=min(scores),
                score_max=max(scores),
                map50=mean_average_precision(ground_truths, remapped, (0.5,)).value,
                map=mean_average_precision(ground_truths, remapped, tuple(np.linspace(0.5, 0.95, 10))).value,
                olrp=lrp.olrp,
                olrp_optimal_threshold=lrp.optimal_threshold,
                precision=point.precision,
                recall=point.recall,
                f1=point.f1,
            )
        )
        logger.info(
            "  %-12s scores in [%.3f, %.3f] -> mAP@.50 = %.6f, oLRP = %s, tau* = %s, F1@%.2f = %.3f",
            name,
            points[-1].score_min,
            points[-1].score_max,
            points[-1].map50,
            f"{lrp.olrp:.3f}" if lrp.olrp is not None else "undefined",
            f"{lrp.optimal_threshold:.3f}" if lrp.optimal_threshold is not None else "undefined",
            score_threshold,
            points[-1].f1,
        )
    return points


@dataclass(frozen=True)
class InterpolationPoint:
    """The envelope's contribution to AP, for categories of one size band."""

    min_objects: int
    max_objects: int
    n_categories: int
    mean_objects: float
    ap50_interpolated: float
    ap50_uninterpolated: float
    bias_mean: float
    bias_std: float
    bias_max: float

    def to_dict(self) -> dict:
        return asdict(self)


DEFAULT_SIZE_BANDS: tuple[tuple[int, int], ...] = (
    (1, 2),
    (3, 5),
    (6, 10),
    (11, 25),
    (26, 50),
    (51, 100),
    (101, 10**6),
)


def interpolation_bias(
    images: Sequence[ImageRecord],
    *,
    seeds: Sequence[int],
    scenario: str = "baseline",
    bands: Sequence[tuple[int, int]] = DEFAULT_SIZE_BANDS,
) -> list[InterpolationPoint]:
    """Measure the monotone envelope\'s contribution to AP against category size.

    mAP is an average of per-category APs, so the question "does interpolation matter
    more when there are few objects?" is a question about *categories*, not about the
    dataset: a category with eight objects has an eight-step precision-recall curve
    whatever the size of the split it sits in.

    For every category, AP@0.50 is computed twice on the same detections -- once with
    the envelope, once without -- and the difference is grouped by how many objects the
    category has. Several seeds are pooled because a single category yields a single
    curve, and the spread between curves of the same size is what says whether a
    difference is real.

    The envelope replaces each precision by the highest precision reached at any greater
    recall, so the difference is non-negative by construction. What it buys is area
    under the dips, and a curve needs enough steps to have dips: the effect is therefore
    not monotone in category size, and the measured shape is reported rather than
    assumed.
    """
    ground_truths = [gt for image in images for gt in image.objects]
    gts_by_category = _group_by_category(ground_truths)

    sizes: list[int] = []
    interpolated: list[float] = []
    raw: list[float] = []
    for seed in seeds:
        detections = generate_detections(images, scenario, seed)
        dts_by_category = _group_by_category(detections)
        for category_id, gts in gts_by_category.items():
            curve = precision_recall_curve(dts_by_category.get(category_id, []), gts, 0.5)
            sizes.append(len(gts))
            interpolated.append(average_precision(curve, "coco"))
            raw.append(average_precision(curve, "none"))

    size = np.array(sizes)
    bias = np.array(interpolated) - np.array(raw)

    points = []
    for low, high in bands:
        in_band = (size >= low) & (size <= high)
        if not in_band.any():
            continue
        points.append(
            InterpolationPoint(
                min_objects=low,
                max_objects=high,
                n_categories=int(in_band.sum()),
                mean_objects=float(size[in_band].mean()),
                ap50_interpolated=float(np.array(interpolated)[in_band].mean()),
                ap50_uninterpolated=float(np.array(raw)[in_band].mean()),
                bias_mean=float(bias[in_band].mean()),
                bias_std=float(bias[in_band].std()),
                bias_max=float(bias[in_band].max()),
            )
        )
        logger.info(
            "  categories with %4d-%-7s objects: n = %4d, AP %.4f -> %.4f, bias = %+.4f +/- %.4f (max %+.4f)",
            low,
            str(high) if high < 10**6 else "more",
            points[-1].n_categories,
            points[-1].ap50_uninterpolated,
            points[-1].ap50_interpolated,
            points[-1].bias_mean,
            points[-1].bias_std,
            points[-1].bias_max,
        )
    return points
