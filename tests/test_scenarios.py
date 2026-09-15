import math

import numpy as np
import pytest

from mapstudy.boxes import iou
from mapstudy.scenarios import (
    SCENARIO_A_SHIFT,
    SCENARIO_B_N_HEDGES,
    SCENARIOS,
    diagonal_shift_iou,
    generate_detections,
)


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_generation_is_reproducible(images, scenario):
    assert generate_detections(images, scenario, seed=7) == generate_detections(images, scenario, seed=7)
    assert generate_detections(images, scenario, seed=7) != generate_detections(images, scenario, seed=8)


def test_diagonal_shift_iou_closed_form():
    assert diagonal_shift_iou(0.0) == 1.0
    assert diagonal_shift_iou(1 - math.sqrt(2 / 3)) == pytest.approx(0.5)
    assert diagonal_shift_iou(SCENARIO_A_SHIFT) == pytest.approx(0.64 / 1.36)


def test_scenario_a_is_just_below_threshold_for_every_object(images):
    detections = generate_detections(images, "a", seed=0)
    gts = [gt for image in images for gt in image.objects]
    assert len(detections) == len(gts)
    for det, gt in zip(detections, gts, strict=True):
        assert iou(det.box, gt.box) == pytest.approx(diagonal_shift_iou(SCENARIO_A_SHIFT))
        assert 0.85 <= det.score <= 0.99


def test_scenario_b_has_one_true_positive_and_sub_threshold_hedges(images):
    detections = generate_detections(images, "b", seed=0)
    gts = [gt for image in images for gt in image.objects]
    per_object = SCENARIO_B_N_HEDGES + 1
    assert len(detections) == per_object * len(gts)

    for index, gt in enumerate(gts):
        true_positive, *hedges = detections[index * per_object : (index + 1) * per_object]
        assert iou(true_positive.box, gt.box) > 0.8
        assert all(iou(h.box, gt.box) < 0.5 for h in hedges)
        assert 0.85 <= true_positive.score <= 0.95
        assert all(h.score < true_positive.score for h in hedges)


def test_baseline_boxes_stay_inside_the_image(images):
    by_id = {image.image_id: image for image in images}
    for det in generate_detections(images, "baseline", seed=0):
        image = by_id[det.image_id]
        x, y, w, h = det.box
        assert w > 0 and h > 0
        assert x >= 0 and y >= 0
        assert x + w <= image.width + 1e-9 and y + h <= image.height + 1e-9


def test_scenario_b_hedge_iou_bound_is_tight():
    # Worst case: offset -0.3 and scale 1.2 on both axes gives 0.81 / 1.63.
    worst = iou((0, 0, 1, 1), (-0.3, -0.3, 1.2, 1.2))
    assert worst == pytest.approx(0.81 / 1.63)
    assert worst < 0.5
    assert np.isclose(worst, 0.497, atol=1e-3)
