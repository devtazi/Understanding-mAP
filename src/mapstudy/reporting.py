"""Human-readable summaries of evaluation reports."""

from __future__ import annotations

from collections.abc import Callable, Sequence

from mapstudy.evaluation import EvaluationReport
from mapstudy.scenarios import SCENARIOS

_ROWS: list[tuple[str, Callable[[EvaluationReport], float | str | None]]] = [
    ("mAP@[.50:.95] (pycocotools)", lambda r: r.coco.map),
    ("mAP@.50 (pycocotools)", lambda r: r.coco.map50),
    ("mAP@.50 (ours, COCO 101-point)", lambda r: r.custom.map50),
    ("mAP@.50 (ours, VOC all-point)", lambda r: r.custom.map50_voc),
    ("mAP@.50 (ours, no interpolation)", lambda r: r.custom.map50_uninterpolated),
    ("oLRP ↓", lambda r: r.lrp.olrp),
    ("oLRP localisation ↓", lambda r: r.lrp.localisation),
    ("oLRP false positive ↓", lambda r: r.lrp.false_positive),
    ("oLRP false negative ↓", lambda r: r.lrp.false_negative),
    ("TIDE AP@.50", lambda r: r.tide.ap50),
    ("TIDE dominant error", lambda r: r.tide.dominant_error or "none"),
    ("Detections / objects", lambda r: f"{r.n_detections} / {r.n_objects}"),
]


def format_results_table(reports: Sequence[EvaluationReport]) -> str:
    """Markdown table with one column per scenario."""
    header = ["Metric", *(SCENARIOS[r.scenario].title.split(":")[0] for r in reports)]
    lines = [_row(header), _row(["---", *(["---:"] * len(reports))])]
    for label, extract in _ROWS:
        lines.append(_row([label, *(_format(extract(r)) for r in reports)]))
    return "\n".join(lines)


def _format(value: float | str | None) -> str:
    if value is None:
        return "undefined"
    if isinstance(value, str):
        return value
    return f"{value:.3f}"


def _row(cells: Sequence[str]) -> str:
    return "| " + " | ".join(cells) + " |"
