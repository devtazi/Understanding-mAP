"""Non-identifiability of mAP: different detectors, one score.

mAP summarises a precision-recall curve as a single number, and many curves share the
same area. This module makes that concrete rather than rhetorical: it takes four
detectors that fail in four mutually exclusive ways — objects missed, spurious boxes,
mislocalised boxes, duplicated boxes — and *calibrates* each one so that all four reach
the same mAP to three decimal places.

What the calibration buys is the right to say the failure is a property of the metric.
Two detectors that happen to land near the same score prove nothing; four detectors
driven to the same score by construction, whose error profiles are known exactly
because they were built that way, show that mAP maps distinct behaviours onto one value
and cannot be inverted.

The same detections are then given to oLRP and TIDE (see :func:`compare_family`), which
is what turns the experiment into a comparison: the question is not whether those
metrics are lower or higher, but whether they separate what mAP merges.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field

from mapstudy.average_precision import mean_average_precision, operating_point
from mapstudy.data import ImageRecord
from mapstudy.evaluation import evaluate_coco, evaluate_lrp, evaluate_tide
from mapstudy.scenarios import (
    EQUIVALENCE_DEFAULT_MEMBERS,
    EQUIVALENCE_FAMILY,
    generate_detections,
)

logger = logging.getLogger(__name__)

#: The family is calibrated to this mAP. Any value both reachable and comparable would
#: do; 0.50 is the one the reference illustration of the problem uses (Oksuz et al.,
#: 2018, Fig. 1), where three detectors with nothing in common all score AP = 0.5.
DEFAULT_TARGET_MAP = 0.50
DEFAULT_TOLERANCE = 5e-4
DEFAULT_MAX_ITERATIONS = 40
DEFAULT_SCORE_THRESHOLD = 0.05


@dataclass(frozen=True)
class Calibration:
    """Knob value that drives one detector to the target mAP, and what it converged to."""

    scenario: str
    parameter: str
    value: float
    map50: float
    iterations: int
    converged: bool


def calibrate(
    images: Sequence[ImageRecord],
    scenario: str,
    *,
    target_map: float = DEFAULT_TARGET_MAP,
    seed: int,
    tolerance: float = DEFAULT_TOLERANCE,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
) -> Calibration:
    """Bisect the scenario's knob until mAP@0.50 reaches ``target_map``.

    ``EQUIVALENCE_FAMILY`` gives the knob and the interval ``(low, high)`` over which
    mAP decreases monotonically from ~1 to below the target, which is what makes plain
    bisection valid here. mAP@0.50 is computed with this project's implementation, which
    the test suite checks against ``pycocotools`` to 1e-9; the reported table re-evaluates
    the calibrated detector with ``pycocotools`` itself.
    """
    parameter, (at_high_map, at_low_map) = EQUIVALENCE_FAMILY[scenario]
    ground_truths = [gt for image in images for gt in image.objects]

    def map_at(value: float) -> float:
        detections = generate_detections(
            images, scenario, seed, per_object_rng=True, **{parameter: value}
        )
        if not detections:
            return 0.0
        return mean_average_precision(ground_truths, detections, (0.5,)).value

    # mAP is a step function of the knob -- it can only change when a detection crosses
    # the threshold -- so the target may fall strictly between two reachable values and
    # plain bisection would never meet the tolerance. The best point seen is kept, and
    # the search reports how close it got.
    low, high = at_high_map, at_low_map
    best = None
    for iteration in range(1, max_iterations + 1):
        value = 0.5 * (low + high)
        achieved = map_at(value)
        if best is None or abs(achieved - target_map) < abs(best.map50 - target_map):
            within_tolerance = abs(achieved - target_map) <= tolerance
            best = Calibration(scenario, parameter, value, achieved, iteration, within_tolerance)
        if best.converged:
            break
        # mAP decreases as the knob moves from ``at_high_map`` towards ``at_low_map``.
        low, high = (low, value) if achieved < target_map else (value, high)

    assert best is not None
    logger.info(
        "  %-13s %s = %.5f -> mAP@.50 = %.4f (%d iterations, |error| = %.1e)",
        scenario, parameter, best.value, best.map50, best.iterations, abs(best.map50 - target_map),
    )
    return best


@dataclass(frozen=True)
class FamilyMember:
    """One calibrated detector, seen by every metric."""

    scenario: str
    parameter: str
    value: float
    n_detections: int
    map50: float
    map: float
    olrp: float | None
    olrp_localisation: float | None
    olrp_false_positive: float | None
    olrp_false_negative: float | None
    olrp_optimal_threshold: float | None
    precision: float
    recall: float
    f1: float
    tide_errors: dict[str, float] = field(default_factory=dict)
    tide_dominant_error: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def compare_family(
    images: Sequence[ImageRecord],
    *,
    target_map: float = DEFAULT_TARGET_MAP,
    seed: int,
    score_threshold: float = DEFAULT_SCORE_THRESHOLD,
    scenarios: Sequence[str] | None = None,
) -> list[FamilyMember]:
    """Calibrate every member of the family to ``target_map``, then evaluate them all.

    Returns one row per detector. Reading the table by column is the point: the mAP
    columns are constant by construction, and every other column is not.
    """
    ground_truths = [gt for image in images for gt in image.objects]

    members = []
    for scenario in scenarios or EQUIVALENCE_DEFAULT_MEMBERS:
        calibration = calibrate(images, scenario, target_map=target_map, seed=seed)
        detections = generate_detections(
            images, scenario, seed, per_object_rng=True, **{calibration.parameter: calibration.value}
        )
        coco = evaluate_coco(images, detections)
        lrp = evaluate_lrp(images, detections)
        tide = evaluate_tide(images, detections, name=scenario)
        point = operating_point(ground_truths, detections, score_threshold)
        members.append(
            FamilyMember(
                scenario=scenario,
                parameter=calibration.parameter,
                value=calibration.value,
                n_detections=len(detections),
                map50=coco.map50,
                map=coco.map,
                olrp=lrp.olrp,
                olrp_localisation=lrp.localisation,
                olrp_false_positive=lrp.false_positive,
                olrp_false_negative=lrp.false_negative,
                olrp_optimal_threshold=lrp.optimal_threshold,
                precision=point.precision,
                recall=point.recall,
                f1=point.f1,
                tide_errors=tide.main_errors,
                tide_dominant_error=tide.dominant_error,
            )
        )
    return members


def discrimination(members: Sequence[FamilyMember]) -> dict[str, float]:
    """Spread of each metric across the family, on a set where mAP is constant by design.

    A metric that cannot tell these four detectors apart has a spread of zero. This is
    the summary number behind the experiment: it says how much information each metric
    retains about *which* detector it is looking at, once the mAP is fixed.
    """

    def spread(values: Sequence[float | None]) -> float:
        defined = [v for v in values if v is not None]
        return float(max(defined) - min(defined)) if defined else 0.0

    return {
        "mAP@0.50": spread([m.map50 for m in members]),
        "mAP@[.50:.95]": spread([m.map for m in members]),
        "oLRP": spread([m.olrp for m in members]),
        "oLRP localisation": spread([m.olrp_localisation for m in members]),
        "oLRP false positive": spread([m.olrp_false_positive for m in members]),
        "oLRP false negative": spread([m.olrp_false_negative for m in members]),
        "TIDE Loc": spread([m.tide_errors.get("Loc", 0.0) for m in members]),
        "TIDE Bkg": spread([m.tide_errors.get("Bkg", 0.0) for m in members]),
        "TIDE Dupe": spread([m.tide_errors.get("Dupe", 0.0) for m in members]),
        "TIDE Miss": spread([m.tide_errors.get("Miss", 0.0) for m in members]),
        "F1 at fixed threshold": spread([m.f1 for m in members]),
    }


@dataclass(frozen=True)
class DuplicationPoint:
    """Every metric at one duplication rate."""

    duplicates_per_object: int
    n_detections: int
    map50: float
    map: float
    olrp: float | None
    olrp_false_positive: float | None
    olrp_optimal_threshold: float | None
    precision: float
    recall: float
    f1: float
    tide_dupe: float | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def duplication_floor(
    images: Sequence[ImageRecord],
    *,
    rates: Sequence[int] = (0, 1, 2, 4, 8, 16, 32, 64),
    seed: int,
    score_threshold: float = DEFAULT_SCORE_THRESHOLD,
    with_tide: bool = True,
) -> list[DuplicationPoint]:
    """Metrics as every object is covered by a growing number of near-identical boxes.

    This is the hedging experiment without its escape hatch. In scenario B the redundant
    boxes score below every accurate one, so a reader can object that the detector is
    merely uncalibrated and a confidence threshold would remove them. Here every copy
    draws its score from the same distribution as the accurate box, so no threshold
    separates them, and the copies are as confident as the detections they surround.

    mAP still barely moves. The reason is structural: the matcher awards the true
    positive to the highest-scoring box covering the object, so the true positive is the
    maximum of ``1 + rate`` draws and rises in the ranking exactly as fast as the false
    positives accumulate below it. mAP therefore has a floor that duplication cannot
    push through, while a fixed-threshold F1 falls towards zero.
    """
    ground_truths = [gt for image in images for gt in image.objects]

    points = []
    for rate in rates:
        detections = generate_detections(
            images, "duplicate", seed, per_object_rng=True, duplicate_rate=float(rate)
        )
        coco = evaluate_coco(images, detections)
        lrp = evaluate_lrp(images, detections)
        point = operating_point(ground_truths, detections, score_threshold)
        tide = evaluate_tide(images, detections, name=f"duplicate={rate}") if with_tide else None
        points.append(
            DuplicationPoint(
                duplicates_per_object=int(rate),
                n_detections=len(detections),
                map50=coco.map50,
                map=coco.map,
                olrp=lrp.olrp,
                olrp_false_positive=lrp.false_positive,
                olrp_optimal_threshold=lrp.optimal_threshold,
                precision=point.precision,
                recall=point.recall,
                f1=point.f1,
                tide_dupe=tide.main_errors.get("Dupe") if tide is not None else None,
            )
        )
        logger.info(
            "  %2d duplicates/object -> %6d detections, mAP@.50 = %.3f, oLRP = %s, F1@%.2f = %.3f",
            rate,
            len(detections),
            points[-1].map50,
            f"{lrp.olrp:.3f}" if lrp.olrp is not None else "undefined",
            score_threshold,
            points[-1].f1,
        )
    return points
