import json

import pytest

from mapstudy.cli import main
from mapstudy.evaluation import evaluate
from mapstudy.reporting import format_results_table
from mapstudy.scenarios import generate_detections


@pytest.fixture(scope="module")
def reports(images):
    return {
        name: evaluate(images, generate_detections(images, name, seed=0), scenario=name, seed=0)
        for name in ("a", "b")
    }


def test_scenario_a_gets_zero_map_and_is_diagnosed_as_localisation(reports):
    report = reports["a"]
    assert report.coco.map50 == 0.0
    assert report.coco.map == 0.0
    assert report.tide.dominant_error == "Loc"
    assert report.lrp.olrp == pytest.approx(1.0)


def test_scenario_b_gets_near_perfect_map50_despite_hedging(reports):
    report = reports["b"]
    assert report.coco.map50 > 0.95
    assert report.n_detections == 5 * report.n_objects
    assert report.custom.map50 == pytest.approx(report.coco.map50)


def test_report_serialises_to_json(reports):
    payload = json.loads(json.dumps(reports["b"].to_dict()))
    assert payload["scenario"] == "b"
    assert set(payload) >= {"coco", "custom", "lrp", "tide", "tide_dominant_error"}


def test_results_table_has_one_column_per_scenario(reports):
    table = format_results_table(list(reports.values()))
    assert table.splitlines()[0] == "| Metric | Scenario A | Scenario B |"
    assert "undefined" in table  # oLRP localisation has no true positive to measure in scenario A


def test_cli_run_writes_results(tmp_path, monkeypatch, image_factory):
    monkeypatch.setattr("mapstudy.cli.load_coco_images", lambda n, **_: image_factory(n))
    main(["run", "--scenario", "A", "--n-images", "10", "--output-dir", str(tmp_path), "--no-tide-plots"])

    assert json.loads((tmp_path / "a.json").read_text())["n_images"] == 10
    assert "Scenario A" in (tmp_path / "summary.md").read_text()
