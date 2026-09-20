"""The non-identifiability experiment: same mAP, different detectors.

These run offline on the synthetic annotations of ``conftest``, so the numbers differ
from the COCO ones in the README; what is asserted is the structure of the result, not
its exact value.
"""

from __future__ import annotations

import pytest

from mapstudy.average_precision import mean_average_precision
from mapstudy.equivalence import calibrate, compare_family, discrimination
from mapstudy.scenarios import EQUIVALENCE_DEFAULT_MEMBERS, generate_detections


@pytest.fixture(scope="module")
def family(images):
    return compare_family(images, target_map=0.5, seed=0)


@pytest.mark.parametrize("scenario", EQUIVALENCE_DEFAULT_MEMBERS)
def test_calibration_reaches_the_target(images, scenario):
    calibration = calibrate(images, scenario, target_map=0.5, seed=0)
    assert calibration.map50 == pytest.approx(0.5, abs=0.01)


@pytest.mark.parametrize("scenario", EQUIVALENCE_DEFAULT_MEMBERS)
def test_knob_is_monotone_in_map(images, scenario):
    """Bisection is only valid because each knob moves mAP in one direction.

    Monotonicity holds with ``per_object_rng``: with the shared stream, changing the
    knob also reshuffles every later object and the relation is no longer monotone.
    """
    from mapstudy.scenarios import EQUIVALENCE_FAMILY

    parameter, (at_high, at_low) = EQUIVALENCE_FAMILY[scenario]
    ground_truths = [gt for image in images for gt in image.objects]

    scores = []
    for fraction in (0.0, 0.25, 0.5, 0.75, 1.0):
        value = at_high + fraction * (at_low - at_high)
        detections = generate_detections(
            images, scenario, 0, per_object_rng=True, **{parameter: value}
        )
        scores.append(mean_average_precision(ground_truths, detections, (0.5,)).value if detections else 0.0)

    assert scores == sorted(scores, reverse=True)


def test_per_object_rng_is_deterministic(images):
    first = generate_detections(images, "ghost", 0, per_object_rng=True, ghost_rate=1.0)
    second = generate_detections(images, "ghost", 0, per_object_rng=True, ghost_rate=1.0)
    assert [(d.image_id, d.box, d.score) for d in first] == [(d.image_id, d.box, d.score) for d in second]


def test_family_shares_one_map(family):
    """The premise of the experiment: mAP cannot tell these detectors apart."""
    spread = max(m.map50 for m in family) - min(m.map50 for m in family)
    assert spread < 0.02


def test_olrp_components_separate_what_map_merges(family):
    """oLRP's decomposition must react where mAP does not."""
    spreads = discrimination(family)
    assert spreads["oLRP false positive"] > 10 * spreads["mAP@0.50"]
    assert spreads["oLRP false negative"] > 10 * spreads["mAP@0.50"]


def test_tide_names_the_error_of_each_detector(family):
    """Each detector was built to fail in one way; TIDE must report that way."""
    dominant = {m.scenario: m.tide_dominant_error for m in family}
    assert dominant["miss"] == "Miss"
    assert dominant["ghost"] == "Bkg"
    assert dominant["mislocalised"] == "Loc"


def test_missed_objects_produce_no_false_positives(family):
    """The 'miss' detector emits perfect boxes only, so its error is purely recall."""
    member = next(m for m in family if m.scenario == "miss")
    assert member.olrp_false_positive == pytest.approx(0.0, abs=1e-6)
    assert member.olrp_localisation == pytest.approx(0.0, abs=1e-6)
    assert member.olrp_false_negative > 0.3


def test_spurious_boxes_produce_no_false_negatives(family):
    """The 'ghost' detector finds every object, so its error is purely precision."""
    member = next(m for m in family if m.scenario == "ghost")
    assert member.olrp_false_positive > 0.3
    assert member.olrp_localisation == pytest.approx(0.0, abs=1e-6)
