"""What AP reads in a confidence score, and what interpolation adds to it."""

from __future__ import annotations

import numpy as np
import pytest

from mapstudy.average_precision import average_precision, mean_average_precision, precision_recall_curve
from mapstudy.invariance import MONOTONE_MAPS, interpolation_bias, remap_scores, score_invariance
from mapstudy.scenarios import generate_detections


@pytest.mark.parametrize("name", list(MONOTONE_MAPS))
def test_map_is_invariant_under_any_monotone_remapping(images, name):
    """AP reads the ranking of scores and nothing else, so it must not move at all."""
    ground_truths = [gt for image in images for gt in image.objects]
    detections = generate_detections(images, "baseline", seed=0)

    reference = mean_average_precision(ground_truths, detections, (0.5,)).value
    remapped = mean_average_precision(
        ground_truths, remap_scores(detections, MONOTONE_MAPS[name]), (0.5,)
    ).value

    assert remapped == reference


def test_remapping_preserves_the_ranking(images):
    detections = generate_detections(images, "baseline", seed=0)
    remapped = remap_scores(detections, MONOTONE_MAPS["cubed"])
    order = np.argsort([-d.score for d in detections], kind="mergesort")
    remapped_order = np.argsort([-d.score for d in remapped], kind="mergesort")
    assert list(order) == list(remapped_order)


def test_squashing_scores_leaves_map_intact_but_destroys_the_operating_point(images):
    """The practical consequence: a detector can be made unusable without mAP noticing."""
    points = {p.transform: p for p in score_invariance(images, seed=0)}
    identity, squashed = points["identity"], points["squashed"]

    assert squashed.map50 == identity.map50
    assert squashed.score_max < 0.15  # every detection now scores below a typical threshold
    assert squashed.olrp_optimal_threshold != identity.olrp_optimal_threshold


def test_interpolation_can_only_raise_ap(images):
    """The monotone envelope replaces each precision by a value at least as large."""
    ground_truths = [gt for image in images for gt in image.objects]
    detections = generate_detections(images, "baseline", seed=0)
    for category in {gt.category_id for gt in ground_truths}:
        gts = [gt for gt in ground_truths if gt.category_id == category]
        dts = [d for d in detections if d.category_id == category]
        curve = precision_recall_curve(dts, gts, 0.5)
        assert average_precision(curve, "coco") >= average_precision(curve, "none") - 1e-12


def test_interpolation_bias_is_non_negative_in_every_size_band(images):
    for point in interpolation_bias(images, seeds=(0, 1, 2)):
        assert point.bias_mean >= 0.0
        assert point.bias_max >= point.bias_mean
