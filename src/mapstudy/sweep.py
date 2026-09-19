"""Parameter sweeps.

A single scenario gives one number; sweeping the parameter that drives it shows the
*shape* of each metric's response. Two sweeps are defined:

- ``shift``: how each metric reacts as localisation degrades continuously. mAP falls off
  a cliff at the IoU threshold while oLRP degrades smoothly.
- ``n_hedges``: how each metric reacts as redundant low-confidence boxes pile up around
  every object. Ranking-based metrics do not react at all.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import asdict, dataclass

import numpy as np

from mapstudy.average_precision import operating_point
from mapstudy.data import ImageRecord
from mapstudy.evaluation import evaluate_coco, evaluate_lrp
from mapstudy.scenarios import diagonal_shift_iou, generate_detections

logger = logging.getLogger(__name__)

DEFAULT_SHIFTS = tuple(np.round(np.linspace(0.0, 0.40, 21), 3))
DEFAULT_HEDGE_COUNTS = tuple(range(0, 13, 2))
DEFAULT_SCORE_THRESHOLD = 0.05


@dataclass(frozen=True)
class SweepPoint:
    """Every metric at one value of the swept parameter."""

    parameter: str
    value: float
    n_detections: int
    map50: float
    map: float
    olrp: float | None
    olrp_localisation: float | None
    precision: float
    recall: float
    f1: float
    prediction_iou: float | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def sweep(
    images: Sequence[ImageRecord],
    scenario: str,
    parameter: str,
    values: Sequence[float],
    *,
    seed: int,
    score_threshold: float = DEFAULT_SCORE_THRESHOLD,
) -> list[SweepPoint]:
    """Evaluate ``scenario`` once per value of ``parameter``."""
    ground_truths = [gt for image in images for gt in image.objects]

    points = []
    for value in values:
        detections = generate_detections(images, scenario, seed, **{parameter: value})
        coco = evaluate_coco(images, detections)
        lrp = evaluate_lrp(images, detections)
        point = operating_point(ground_truths, detections, score_threshold)
        points.append(
            SweepPoint(
                parameter=parameter,
                value=float(value),
                n_detections=len(detections),
                map50=coco.map50,
                map=coco.map,
                olrp=lrp.olrp,
                olrp_localisation=lrp.localisation,
                precision=point.precision,
                recall=point.recall,
                f1=point.f1,
                prediction_iou=diagonal_shift_iou(value) if parameter == "shift" else None,
            )
        )
        logger.info(
            "  %s = %-5s -> mAP@.50 = %.3f, oLRP = %s, F1@%.2f = %.3f",
            parameter,
            f"{value:g}",
            points[-1].map50,
            f"{lrp.olrp:.3f}" if lrp.olrp is not None else "undefined",
            score_threshold,
            points[-1].f1,
        )
    return points
