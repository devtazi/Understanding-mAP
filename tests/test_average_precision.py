import numpy as np
import pytest

from mapstudy.average_precision import (
    COCO_IOU_THRESHOLDS,
    PrecisionRecallCurve,
    average_precision,
    mean_average_precision,
    precision_recall_curve,
)
from mapstudy.data import Detection, GroundTruth
from mapstudy.evaluation import evaluate_coco
from mapstudy.scenarios import SCENARIOS, generate_detections


def gt(image_id, box, category_id=0):
    return GroundTruth(image_id, category_id, box, box[2] * box[3])


def det(image_id, box, score, category_id=0):
    return Detection(image_id, category_id, box, score)


def curve(is_tp, n_gts):
    tp = np.cumsum(is_tp)
    fp = np.cumsum(~np.asarray(is_tp))
    return PrecisionRecallCurve(np.zeros(len(is_tp)), tp / (tp + fp), tp / n_gts, n_gts)


def test_perfect_detector_scores_one():
    gts = [gt(1, (0, 0, 10, 10)), gt(1, (50, 50, 10, 10))]
    dts = [det(1, g.box, 0.9) for g in gts]
    for interpolation in ("coco", "voc", "none"):
        assert mean_average_precision(gts, dts, interpolation=interpolation).value == pytest.approx(1.0)


def test_detector_without_true_positive_scores_zero():
    gts = [gt(1, (0, 0, 10, 10))]
    dts = [det(1, (100, 100, 10, 10), 0.9)]
    assert mean_average_precision(gts, dts).value == 0.0


def test_duplicate_detections_are_false_positives():
    gts = [gt(1, (0, 0, 10, 10))]
    dts = [det(1, (0, 0, 10, 10), 0.9), det(1, (0, 0, 10, 10), 0.8)]
    pr = precision_recall_curve(dts, gts, iou_threshold=0.5)
    np.testing.assert_allclose(pr.precision, [1.0, 0.5])
    np.testing.assert_allclose(pr.recall, [1.0, 1.0])


def test_each_detection_matches_the_best_unmatched_ground_truth():
    gts = [gt(1, (0, 0, 10, 10)), gt(1, (4, 0, 10, 10))]
    # The first detection overlaps both; it must take the better one and leave the other.
    dts = [det(1, (3, 0, 10, 10), 0.9), det(1, (0, 0, 10, 10), 0.8)]
    pr = precision_recall_curve(dts, gts, iou_threshold=0.5)
    np.testing.assert_allclose(pr.recall, [0.5, 1.0])


def test_interpolation_schemes_on_a_hand_computed_curve():
    # Ranked detections: TP, FP, TP out of 2 ground truths.
    # Raw precision is 1, 1/2, 2/3 at recall 1/2, 1/2, 1.
    pr = curve([True, False, True], n_gts=2)
    assert average_precision(pr, "none") == pytest.approx(0.5 * 1 + 0.5 * 2 / 3)
    assert average_precision(pr, "voc") == pytest.approx(0.5 * 1 + 0.5 * 2 / 3)
    # COCO: recall levels 0..0.5 (51 points) at precision 1, 0.51..1 (50 points) at 2/3.
    assert average_precision(pr, "coco") == pytest.approx((51 * 1 + 50 * 2 / 3) / 101)


def test_interpolation_fills_precision_dips():
    # TP, FP, FP, TP, TP out of 3: raw precision dips to 1/2 then recovers to 3/5.
    pr = curve([True, False, False, True, True], n_gts=3)
    assert average_precision(pr, "voc") > average_precision(pr, "none")


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_matches_pycocotools(images, scenario):
    detections = generate_detections(images, scenario, seed=3)
    gts = [g for image in images for g in image.objects]
    reference = evaluate_coco(images, detections)

    assert mean_average_precision(gts, detections).value == pytest.approx(reference.map50, abs=1e-9)
    assert mean_average_precision(gts, detections, COCO_IOU_THRESHOLDS).value == pytest.approx(
        reference.map, abs=1e-9
    )
