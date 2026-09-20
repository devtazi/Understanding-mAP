"""Evaluation of a set of detections with every metric used in the study.

- mAP from the reference ``pycocotools`` implementation;
- mAP from this project's own implementation, with each interpolation scheme;
- optimal LRP and its components (Oksuz et al., 2018);
- TIDE error breakdown (Bolya et al., 2020).

All values are fractions in ``[0, 1]``. TIDE reports in percentage points; its
values are rescaled here for consistency.
"""

from __future__ import annotations

import contextlib
import io
import logging
import math
import warnings
from collections.abc import Iterator, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from tidecv import TIDE
from tidecv import Data as TideData

from mapstudy.average_precision import COCO_IOU_THRESHOLDS, mean_average_precision
from mapstudy.data import Detection, ImageRecord
from mapstudy.third_party.cocoeval_lrp import COCOeval as LRPCOCOeval

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CocoMetrics:
    map: float
    map50: float
    map75: float


@dataclass(frozen=True)
class CustomMetrics:
    """mAP@0.50 from this project's implementation, for each interpolation scheme."""

    map: float
    map50: float
    map50_voc: float
    map50_uninterpolated: float


@dataclass(frozen=True)
class LrpMetrics:
    """Optimal LRP at IoU 0.5. Lower is better. ``None`` when undefined (no true positive).

    ``optimal_threshold`` is the confidence threshold tau* at which the minimum is
    attained. oLRP is a *minimum over thresholds*, so the score alone hides where that
    minimum sits; tau* is the part of the metric that says which detections it kept.
    Reporting it is what lets oLRP answer a question mAP cannot: at what confidence
    should this detector be deployed.
    """

    olrp: float | None
    localisation: float | None
    false_positive: float | None
    false_negative: float | None
    optimal_threshold: float | None = None


@dataclass(frozen=True)
class TideMetrics:
    """TIDE AP@0.50 and the AP gained by fixing each error type."""

    ap50: float
    main_errors: dict[str, float]
    special_errors: dict[str, float]

    @property
    def dominant_error(self) -> str | None:
        """Error type whose fix recovers the most AP, or ``None`` if no fix recovers any."""
        worst = max(self.main_errors, key=self.main_errors.get)
        return worst if self.main_errors[worst] > 0 else None


@dataclass(frozen=True)
class EvaluationReport:
    scenario: str
    seed: int
    n_images: int
    n_objects: int
    n_detections: int
    coco: CocoMetrics
    custom: CustomMetrics
    lrp: LrpMetrics
    tide: TideMetrics

    def to_dict(self) -> dict:
        return asdict(self) | {"tide_dominant_error": self.tide.dominant_error}


def evaluate(
    images: Sequence[ImageRecord],
    detections: Sequence[Detection],
    *,
    scenario: str,
    seed: int,
    tide_plot_dir: Path | None = None,
) -> EvaluationReport:
    """Run every metric on the same ground truth and detections."""
    if not detections:
        raise ValueError("Cannot evaluate an empty set of detections.")

    return EvaluationReport(
        scenario=scenario,
        seed=seed,
        n_images=len(images),
        n_objects=sum(len(image.objects) for image in images),
        n_detections=len(detections),
        coco=evaluate_coco(images, detections),
        custom=evaluate_custom(images, detections),
        lrp=evaluate_lrp(images, detections),
        tide=evaluate_tide(images, detections, name=scenario, plot_dir=tide_plot_dir),
    )


def evaluate_coco(images: Sequence[ImageRecord], detections: Sequence[Detection]) -> CocoMetrics:
    coco_gt, coco_dt = _coco_datasets(images, detections)
    with _silenced():
        evaluator = COCOeval(coco_gt, coco_dt, iouType="bbox")
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()
    return CocoMetrics(*(float(v) for v in evaluator.stats[:3]))


def evaluate_custom(images: Sequence[ImageRecord], detections: Sequence[Detection]) -> CustomMetrics:
    gts = [gt for image in images for gt in image.objects]
    return CustomMetrics(
        map=mean_average_precision(gts, detections, COCO_IOU_THRESHOLDS).value,
        map50=mean_average_precision(gts, detections, interpolation="coco").value,
        map50_voc=mean_average_precision(gts, detections, interpolation="voc").value,
        map50_uninterpolated=mean_average_precision(gts, detections, interpolation="none").value,
    )


def evaluate_lrp(images: Sequence[ImageRecord], detections: Sequence[Detection]) -> LrpMetrics:
    coco_gt, coco_dt = _coco_datasets(images, detections)
    with _silenced():
        evaluator = LRPCOCOeval(coco_gt, coco_dt, iouType="bbox")
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()
    # The evaluator reports -1 for an undefined metric, e.g. localisation error without TP.
    olrp, loc, fp, fn = (None if v < 0 or math.isnan(v) else float(v) for v in evaluator.stats[12:16])
    return LrpMetrics(olrp, loc, fp, fn, _optimal_threshold(evaluator))


def _optimal_threshold(evaluator: LRPCOCOeval) -> float | None:
    """Mean over categories of tau*, the confidence threshold minimising LRP.

    ``accumulate`` stores one threshold per category in ``eval['lrp_opt_thr']``. The
    array is initialised to -1 and set to NaN for a category with no true positive, so
    both sentinels are skipped and only real thresholds are averaged.
    """
    thresholds = evaluator.eval.get("lrp_opt_thr")
    if thresholds is None:
        return None
    defined = thresholds[np.isfinite(thresholds) & (thresholds >= 0)]
    return float(defined.mean()) if defined.size else None


def evaluate_tide(
    images: Sequence[ImageRecord],
    detections: Sequence[Detection],
    *,
    name: str,
    plot_dir: Path | None = None,
) -> TideMetrics:
    ground_truth = TideData("ground truth")
    for image in images:
        for gt in image.objects:
            ground_truth.add_ground_truth(gt.image_id, gt.category_id, list(gt.box))

    predictions = TideData(name)
    for det in detections:
        predictions.add_detection(det.image_id, det.category_id, det.score, list(det.box))

    tide = TIDE()
    with _silenced():
        run = tide.evaluate(ground_truth, predictions, mode=TIDE.BOX)
        errors = tide.get_all_errors()
    metrics = TideMetrics(
        ap50=run.ap / 100,
        main_errors={k: v / 100 for k, v in errors["main"][name].items()},
        special_errors={k: v / 100 for k, v in errors["special"][name].items()},
    )

    if plot_dir is not None:
        # TIDE's summary plot divides by the total main error and crashes when it is zero.
        if metrics.dominant_error is None:
            logger.info("No TIDE main error for %r: summary plot skipped.", name)
        else:
            plot_dir.mkdir(parents=True, exist_ok=True)
            with _silenced():
                tide.plot(out_dir=str(plot_dir))

    return metrics


def _coco_datasets(images: Sequence[ImageRecord], detections: Sequence[Detection]) -> tuple[COCO, COCO]:
    """Build in-memory pycocotools datasets. Fresh objects each time, as evaluators mutate them."""
    annotations = [
        {
            "id": index,
            "image_id": gt.image_id,
            "category_id": gt.category_id,
            "bbox": list(gt.box),
            "area": gt.area,
            "iscrowd": 0,
        }
        for index, gt in enumerate((gt for image in images for gt in image.objects), start=1)
    ]
    results = [
        {"image_id": d.image_id, "category_id": d.category_id, "bbox": list(d.box), "score": d.score}
        for d in detections
    ]
    categories = sorted({a["category_id"] for a in annotations} | {r["category_id"] for r in results})

    with _silenced():
        coco_gt = COCO()
        coco_gt.dataset = {
            "images": [{"id": i.image_id, "width": i.width, "height": i.height} for i in images],
            "annotations": annotations,
            "categories": [{"id": c} for c in categories],
        }
        coco_gt.createIndex()
        coco_dt = coco_gt.loadRes(results)
    return coco_gt, coco_dt


@contextlib.contextmanager
def _silenced() -> Iterator[None]:
    """Hide the progress output and deprecation warnings of pycocotools, TIDE and its plotting stack."""
    with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_SILENT)  # TIDE saves figures with cv2
        yield
