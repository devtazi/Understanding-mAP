"""Parameter sweeps.

A single scenario gives one number; sweeping the parameter that drives it shows the
*shape* of each metric's response. Three sweeps are defined:

- ``shift``: how each metric reacts as localisation degrades continuously. mAP falls off
  a cliff at the IoU threshold while oLRP degrades smoothly.
- ``n_hedges``: how each metric reacts as redundant low-confidence boxes pile up around
  every object. Ranking-based metrics do not react at all.
- ``hedge_max_score``: how each metric reacts as those redundant boxes stop being cleanly
  separable by confidence, which is what a ranking-based metric actually depends on.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import asdict, dataclass

import numpy as np

from mapstudy.average_precision import operating_point
from mapstudy.data import ImageRecord
from mapstudy.evaluation import evaluate_coco, evaluate_lrp, evaluate_tide
from mapstudy.scenarios import diagonal_shift_iou, generate_detections

logger = logging.getLogger(__name__)

DEFAULT_SHIFTS = tuple(np.round(np.linspace(0.0, 0.60, 25), 3))
DEFAULT_HEDGE_COUNTS = tuple(range(0, 13, 2))
DEFAULT_HEDGE_MAX_SCORES = tuple(np.round(np.linspace(0.40, 0.99, 13), 3))
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
    tide_errors: dict[str, float] | None = None
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
    with_tide: bool = True,
) -> list[SweepPoint]:
    """Evaluate ``scenario`` once per value of ``parameter``."""
    ground_truths = [gt for image in images for gt in image.objects]

    points = []
    for value in values:
        detections = generate_detections(images, scenario, seed, **{parameter: value})
        coco = evaluate_coco(images, detections)
        lrp = evaluate_lrp(images, detections)
        point = operating_point(ground_truths, detections, score_threshold)
        tide = evaluate_tide(images, detections, name=f"{parameter}={value:g}") if with_tide else None
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
                tide_errors=tide.main_errors if tide is not None else None,
                prediction_iou=diagonal_shift_iou(value) if parameter == "shift" else None,
            )
        )
        logger.info(
            "  %s = %-5s -> mAP@.50 = %.3f, oLRP = %s, F1@%.2f = %.3f, TIDE: %s",
            parameter,
            f"{value:g}",
            points[-1].map50,
            f"{lrp.olrp:.3f}" if lrp.olrp is not None else "undefined",
            score_threshold,
            points[-1].f1,
            tide.dominant_error or "no error to fix" if tide is not None else "skipped",
        )
    return points
