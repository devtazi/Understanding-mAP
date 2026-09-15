import numpy as np
import pytest

from mapstudy.boxes import intersection_area, iou, iou_matrix


def test_identical_boxes_have_iou_one():
    assert iou((10, 20, 30, 40), (10, 20, 30, 40)) == pytest.approx(1.0)


def test_disjoint_and_touching_boxes_have_iou_zero():
    assert iou((0, 0, 10, 10), (50, 50, 10, 10)) == 0.0
    assert iou((0, 0, 10, 10), (10, 0, 10, 10)) == 0.0


def test_partial_overlap_matches_hand_computation():
    # Overlap is 5 x 10 = 50, union is 100 + 100 - 50 = 150.
    assert intersection_area((0, 0, 10, 10), (5, 0, 10, 10)) == 50
    assert iou((0, 0, 10, 10), (5, 0, 10, 10)) == pytest.approx(1 / 3)


def test_contained_box():
    assert iou((0, 0, 10, 10), (0, 0, 5, 5)) == pytest.approx(0.25)


def test_degenerate_boxes_do_not_divide_by_zero():
    assert iou((0, 0, 0, 0), (0, 0, 0, 0)) == 0.0


def test_iou_matrix_agrees_with_scalar_iou():
    rng = np.random.default_rng(0)
    a = rng.uniform(0, 100, size=(7, 4))
    b = rng.uniform(0, 100, size=(5, 4))
    expected = np.array([[iou(x, y) for y in b] for x in a])
    np.testing.assert_allclose(iou_matrix(a, b), expected)
