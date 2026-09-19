import pytest

from mapstudy.average_precision import operating_point
from mapstudy.data import Detection, GroundTruth
from mapstudy.scenarios import diagonal_shift_iou, generate_detections
from mapstudy.sweep import sweep


def gt(image_id, box, category_id=0):
    return GroundTruth(image_id, category_id, box, box[2] * box[3])


def det(image_id, box, score, category_id=0):
    return Detection(image_id, category_id, box, score)


def test_operating_point_counts_and_rates():
    gts = [gt(1, (0, 0, 10, 10)), gt(1, (50, 50, 10, 10)), gt(2, (0, 0, 10, 10))]
    dts = [
        det(1, (0, 0, 10, 10), 0.9),  # true positive
        det(1, (0, 0, 10, 10), 0.8),  # duplicate, false positive
        det(1, (200, 200, 10, 10), 0.7),  # background, false positive
        det(2, (0, 0, 10, 10), 0.01),  # below the threshold, ignored
    ]
    point = operating_point(gts, dts, score_threshold=0.05)

    assert (point.true_positives, point.false_positives, point.false_negatives) == (1, 2, 2)
    assert point.precision == pytest.approx(1 / 3)
    assert point.recall == pytest.approx(1 / 3)
    assert point.f1 == pytest.approx(1 / 3)


def test_operating_point_ignores_ranking_unlike_average_precision(images):
    """Hedges are invisible to mAP but must lower the F1 of a deployed detector."""
    without = generate_detections(images, "b", seed=0, n_hedges=0)
    with_hedges = generate_detections(images, "b", seed=0, n_hedges=4)
    gts = [g for image in images for g in image.objects]

    assert operating_point(gts, without, 0.05).f1 == pytest.approx(1.0)
    assert operating_point(gts, with_hedges, 0.05).f1 < 0.4


def test_shift_sweep_shows_the_threshold_cliff(images):
    # 0.18 gives IoU just above 0.5, 0.20 just below.
    below, above = 0.18, 0.20
    assert diagonal_shift_iou(below) > 0.5 > diagonal_shift_iou(above)

    points = sweep(images, "a", "shift", [below, above], seed=0)
    assert points[0].map50 > 0.95
    assert points[1].map50 < 0.05
    # oLRP moves continuously across the same step instead of collapsing.
    assert points[1].olrp - points[0].olrp < 0.2


def test_hedging_sweep_leaves_ranking_metrics_untouched(images):
    points = sweep(images, "b", "n_hedges", [0, 4, 8], seed=0)

    assert [p.map50 for p in points] == pytest.approx([points[0].map50] * len(points))
    assert [p.olrp for p in points] == pytest.approx([points[0].olrp] * len(points))
    assert [p.f1 for p in points] == sorted((p.f1 for p in points), reverse=True)
    assert [p.n_detections for p in points] == sorted(p.n_detections for p in points)


def test_sweep_point_serialises(images):
    point = sweep(images, "a", "shift", [0.2], seed=0)[0]
    payload = point.to_dict()
    assert payload["parameter"] == "shift"
    assert payload["prediction_iou"] == pytest.approx(diagonal_shift_iou(0.2))
