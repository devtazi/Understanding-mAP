"""Synthetic detectors.

Instead of running a trained model, each scenario derives predictions directly from
the ground-truth boxes. This isolates the behaviour of the *metric* from any
model-specific noise: every failure observed downstream is caused by how the metric
scores a known, controlled prediction pattern.

Scenario generators do not clip boxes to the image: clipping would change the IoU
that scenarios A and B are designed to control. Evaluation tools accept such boxes.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import partial

import numpy as np

from mapstudy.data import Detection, GroundTruth, ImageRecord

DetectorFn = Callable[[GroundTruth, ImageRecord, np.random.Generator], list[Detection]]

BASELINE_JITTER = 0.25

SCENARIO_A_SHIFT = 0.20
SCENARIO_A_SCORE_RANGE = (0.85, 0.99)

SCENARIO_B_TP_SHIFT = 0.05
SCENARIO_B_TP_SCORE_RANGE = (0.85, 0.95)
SCENARIO_B_N_HEDGES = 4
SCENARIO_B_HEDGE_OFFSET_RANGE = (0.30, 0.70)
SCENARIO_B_HEDGE_SCALE_RANGE = (0.80, 1.20)
SCENARIO_B_HEDGE_SCORE_RANGE = (0.10, 0.40)


def diagonal_shift_iou(shift: float) -> float:
    """IoU between a box and a same-size copy shifted by ``shift`` of its size on both axes.

    With an overlap ratio ``r = 1 - shift`` on each axis, ``IoU = r² / (2 - r²)``. It
    does not depend on the box size, which is what makes scenario A exact.
    """
    overlap = (1.0 - shift) ** 2
    return overlap / (2.0 - overlap)


def noisy_detector(
    gt: GroundTruth,
    image: ImageRecord,
    rng: np.random.Generator,
    *,
    jitter: float = BASELINE_JITTER,
) -> list[Detection]:
    """Baseline: random jitter of position and size, with an uninformative score.

    Each coordinate is perturbed by up to ``jitter`` of the box size and the result is
    clipped to the image. Scores are drawn uniformly, independently of IoU.
    """
    x, y, w, h = gt.box
    dx, dy, dw, dh = rng.uniform(-jitter, jitter, size=4) * (w, h, w, h)

    x1 = np.clip(x + dx, 0, image.width)
    y1 = np.clip(y + dy, 0, image.height)
    x2 = np.clip(x + dx + max(1.0, w + dw), 0, image.width)
    y2 = np.clip(y + dy + max(1.0, h + dh), 0, image.height)
    if x2 <= x1 or y2 <= y1:
        return []

    box = (float(x1), float(y1), float(x2 - x1), float(y2 - y1))
    return [Detection(gt.image_id, gt.category_id, box, float(rng.uniform(0.1, 1.0)))]


def sub_threshold_detector(
    gt: GroundTruth,
    image: ImageRecord,
    rng: np.random.Generator,
    *,
    shift: float = SCENARIO_A_SHIFT,
) -> list[Detection]:
    """Scenario A: confident predictions that sit just below the IoU = 0.5 threshold.

    Every box keeps its size and is shifted by ``shift`` of its width and height towards
    the bottom-right, giving ``IoU = diagonal_shift_iou(shift)`` for every object,
    ``≈ 0.471`` at the default 0.2. The shift giving exactly 0.5 is ``1 - sqrt(2/3) ≈ 0.184``.
    Sweeping ``shift`` is what draws the mAP cliff (see :mod:`mapstudy.sweep`).
    """
    x, y, w, h = gt.box
    box = (x + shift * w, y + shift * h, w, h)
    score = float(rng.uniform(*SCENARIO_A_SCORE_RANGE))
    return [Detection(gt.image_id, gt.category_id, box, score)]


def hedging_detector(
    gt: GroundTruth,
    image: ImageRecord,
    rng: np.random.Generator,
    *,
    n_hedges: int = SCENARIO_B_N_HEDGES,
    hedge_max_score: float = SCENARIO_B_HEDGE_SCORE_RANGE[1],
) -> list[Detection]:
    """Scenario B: one accurate box plus ``n_hedges`` low-confidence "hedging" boxes.

    The accurate box is shifted by ``SCENARIO_B_TP_SHIFT`` (IoU ≈ 0.82) and receives a
    high score. Each hedge is displaced by 30-70% of the box size in a random direction
    on both axes and rescaled by 0.8-1.2. Over that parameter range the IoU with the
    target object is at most 0.497, so hedges can never be true positives for it.

    Hedge scores are drawn from ``U(0.10, hedge_max_score)``. As long as that range stays
    below the accurate boxes' scores, the two populations are separable by a confidence
    threshold; raising ``hedge_max_score`` past 0.85 makes them overlap, which is the
    situation ranking-based metrics can no longer hide (see :mod:`mapstudy.sweep`).
    """
    x, y, w, h = gt.box
    tp_box = (x + SCENARIO_B_TP_SHIFT * w, y + SCENARIO_B_TP_SHIFT * h, w, h)
    detections = [
        Detection(gt.image_id, gt.category_id, tp_box, float(rng.uniform(*SCENARIO_B_TP_SCORE_RANGE)))
    ]

    for _ in range(n_hedges):
        offset = rng.uniform(*SCENARIO_B_HEDGE_OFFSET_RANGE, size=2) * rng.choice((-1, 1), size=2)
        scale = rng.uniform(*SCENARIO_B_HEDGE_SCALE_RANGE, size=2)
        box = (x + offset[0] * w, y + offset[1] * h, scale[0] * w, scale[1] * h)
        score = float(rng.uniform(SCENARIO_B_HEDGE_SCORE_RANGE[0], hedge_max_score))
        detections.append(Detection(gt.image_id, gt.category_id, box, score))

    return detections


@dataclass(frozen=True)
class Scenario:
    name: str
    title: str
    detector: DetectorFn


SCENARIOS: dict[str, Scenario] = {
    "baseline": Scenario("baseline", "Baseline: noisy detector", noisy_detector),
    "a": Scenario("a", "Scenario A: sub-threshold localisation", sub_threshold_detector),
    "b": Scenario("b", "Scenario B: spatial hedging", hedging_detector),
}


def generate_detections(
    images: Sequence[ImageRecord],
    scenario: str,
    seed: int,
    *,
    per_object_rng: bool = False,
    **parameters: float,
) -> list[Detection]:
    """Apply a scenario to every object of every image, reproducibly for a given seed.

    Keyword parameters are forwarded to the detector, which is how sweeps vary a single
    knob (``shift=0.3`` for scenario A, ``n_hedges=10`` for scenario B).

    With ``per_object_rng``, each object is given its own generator seeded from
    ``(seed, image_id, index)`` instead of drawing from one shared stream. The shared
    stream is fine for a single evaluation, but it makes a detector's output depend on
    the knob in a second, unwanted way: changing the knob changes how many draws each
    object consumes, so every later object gets different randomness and the metric
    stops being a monotone function of the knob. Per-object generators remove that
    coupling, which is what lets :mod:`mapstudy.equivalence` bisect a knob to a target
    mAP. Existing scenarios keep the shared stream so their published results are
    unchanged.
    """
    detector = (
        partial(SCENARIOS[scenario].detector, **parameters) if parameters else (SCENARIOS[scenario].detector)
    )
    if not per_object_rng:
        rng = np.random.default_rng(seed)
        return [
            detection
            for image in images
            for gt in image.objects
            for detection in detector(gt, image, rng)
        ]

    return [
        detection
        for image in images
        for index, gt in enumerate(image.objects)
        for detection in detector(gt, image, np.random.default_rng((seed, image.image_id, index)))
    ]


# --- Equivalence family -------------------------------------------------------
#
# Four detectors whose single knob controls mAP monotonically, each failing in one
# pure way. Calibrating the knobs so that all four reach the *same* mAP is what makes
# the non-identifiability of mAP measurable (see :mod:`mapstudy.equivalence`).
#
# Every box that is meant to be a true positive is the ground-truth box itself, so its
# IoU is exactly 1.0 and no localisation error contaminates the comparison.

EQUIVALENCE_SCORE_RANGE = (0.05, 1.00)
EQUIVALENCE_GHOST_OFFSET = 3.0
EQUIVALENCE_DUPLICATE_SHIFT = 0.05


def _count(rate: float, rng: np.random.Generator) -> int:
    """Integer count with the fractional part realised as a Bernoulli draw.

    Lets a knob such as "spurious boxes per object" vary continuously, which is what
    the calibration bisection needs.
    """
    whole = int(np.floor(rate))
    return whole + int(rng.random() < rate - whole)


def missing_detector(
    gt: GroundTruth, image: ImageRecord, rng: np.random.Generator, *, detect_rate: float = 0.5
) -> list[Detection]:
    """Perfect boxes on a fraction ``detect_rate`` of objects, nothing on the others.

    Precision is 1 at every rank and recall stops at ``detect_rate``, so AP ≈
    ``detect_rate``. Pure false-negative failure: what it reports, it reports perfectly.
    """
    if rng.random() >= detect_rate:
        return []
    score = float(rng.uniform(*EQUIVALENCE_SCORE_RANGE))
    return [Detection(gt.image_id, gt.category_id, gt.box, score)]


def ghost_detector(
    gt: GroundTruth, image: ImageRecord, rng: np.random.Generator, *, ghost_rate: float = 1.0
) -> list[Detection]:
    """Perfect boxes on every object, plus ``ghost_rate`` spurious boxes per object.

    Spurious boxes are placed ``EQUIVALENCE_GHOST_OFFSET`` box-sizes away, so they
    overlap nothing, and their scores are drawn from the same distribution as the
    accurate ones: the two populations interleave, precision sits at ``1/(1+ghost_rate)``
    at every rank and AP ≈ ``1/(1 + ghost_rate)``. Pure false-positive failure, with
    nothing missed and nothing mislocalised.
    """
    x, y, w, h = gt.box
    score = float(rng.uniform(*EQUIVALENCE_SCORE_RANGE))
    detections = [Detection(gt.image_id, gt.category_id, gt.box, score)]

    for _ in range(_count(ghost_rate, rng)):
        direction = rng.choice((-1.0, 1.0), size=2)
        offset = EQUIVALENCE_GHOST_OFFSET
        box = (x + direction[0] * offset * w, y + direction[1] * offset * h, w, h)
        detections.append(
            Detection(gt.image_id, gt.category_id, box, float(rng.uniform(*EQUIVALENCE_SCORE_RANGE)))
        )
    return detections


def mislocalised_detector(
    gt: GroundTruth, image: ImageRecord, rng: np.random.Generator, *, error_rate: float = 0.3
) -> list[Detection]:
    """Perfect boxes, except on a fraction ``error_rate`` of objects shifted below the threshold.

    A shifted box is a false positive *and* leaves its object unmatched, so precision and
    recall both fall to ``1 - error_rate`` and AP ≈ ``(1 - error_rate)²``. Pure
    localisation failure: every object is found, some boxes are simply not tight enough.
    """
    x, y, w, h = gt.box
    mislocalised = rng.random() < error_rate
    box = (x + SCENARIO_A_SHIFT * w, y + SCENARIO_A_SHIFT * h, w, h) if mislocalised else gt.box
    score = float(rng.uniform(*EQUIVALENCE_SCORE_RANGE))
    return [Detection(gt.image_id, gt.category_id, box, score)]


def duplicate_detector(
    gt: GroundTruth, image: ImageRecord, rng: np.random.Generator, *, duplicate_rate: float = 1.0
) -> list[Detection]:
    """Perfect boxes on every object, plus ``duplicate_rate`` near-copies of each.

    A copy overlaps its object well above the IoU threshold but the object is already
    matched, so it counts as a false positive exactly like a spurious box: AP ≈
    ``1/(1 + duplicate_rate)``, the same law as :func:`ghost_detector`. The two differ
    only in *which kind* of false positive they emit, which is the distinction mAP and
    oLRP cannot make and TIDE can.
    """
    x, y, w, h = gt.box
    score = float(rng.uniform(*EQUIVALENCE_SCORE_RANGE))
    detections = [Detection(gt.image_id, gt.category_id, gt.box, score)]

    for _ in range(_count(duplicate_rate, rng)):
        offset = rng.uniform(-EQUIVALENCE_DUPLICATE_SHIFT, EQUIVALENCE_DUPLICATE_SHIFT, size=2)
        box = (x + offset[0] * w, y + offset[1] * h, w, h)
        detections.append(
            Detection(gt.image_id, gt.category_id, box, float(rng.uniform(*EQUIVALENCE_SCORE_RANGE)))
        )
    return detections


SCENARIOS.update(
    {
        "miss": Scenario("miss", "Missed objects only", missing_detector),
        "ghost": Scenario("ghost", "Spurious boxes only", ghost_detector),
        "mislocalised": Scenario("mislocalised", "Mislocalised boxes only", mislocalised_detector),
        "duplicate": Scenario("duplicate", "Duplicate boxes only", duplicate_detector),
    }
)

#: Knob controlling mAP for each member of the equivalence family, and the range the
#: calibration searches. Each knob is monotone decreasing in mAP over its range.
#:
#: ``duplicate`` is deliberately excluded from :data:`EQUIVALENCE_DEFAULT_MEMBERS`: its
#: mAP cannot be driven down to the common target at all. Duplicating a box is close to
#: free under mAP, because the matcher keeps the highest-scoring copy as the true
#: positive, so the true positive is the maximum of the copies\' scores and is pushed to
#: the top of the ranking. That saturation is itself a result (see
#: :func:`mapstudy.equivalence.duplication_floor`), not a calibration failure.
EQUIVALENCE_FAMILY: dict[str, tuple[str, tuple[float, float]]] = {
    "miss": ("detect_rate", (1.0, 0.0)),
    "ghost": ("ghost_rate", (0.0, 8.0)),
    "mislocalised": ("error_rate", (0.0, 1.0)),
    "duplicate": ("duplicate_rate", (0.0, 8.0)),
}

#: Members that can all be calibrated to a common mAP, one per pure error type.
EQUIVALENCE_DEFAULT_MEMBERS: tuple[str, ...] = ("miss", "ghost", "mislocalised")
