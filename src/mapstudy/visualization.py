"""Figures: what each synthetic detector predicts, and how the metrics respond to it."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

from mapstudy.boxes import Box, iou
from mapstudy.data import Detection, ImageRecord
from mapstudy.sweep import SweepPoint

GT_COLOR = "#22c55e"
MATCHED_COLOR = "#3b82f6"
UNMATCHED_COLOR = "#ef4444"

METRIC_STYLES = {
    "map50": ("mAP@0.50", "#2563eb", "-"),
    "map": ("mAP@[0.50:0.95]", "#7c3aed", "-"),
    "lrp_quality": ("1 − oLRP", "#059669", "-"),
    "f1": ("F1 at score ≥ 0.05", "#ea580c", "--"),
}


def plot_detections(
    image: ImageRecord,
    detections: Sequence[Detection],
    *,
    title: str,
    iou_threshold: float = 0.5,
    output: Path | None = None,
) -> Figure:
    """Draw ground truths and detections on an image, optionally saving the figure.

    Each detection is labelled with its best IoU against a ground truth of the same
    category and its score, and coloured by whether that IoU reaches the threshold.
    The view is cropped to the image, so boxes extending past its border are cut.
    """
    pixels = image.load_image()
    fig, ax = plt.subplots(figsize=(10, 10 * pixels.height / pixels.width))
    ax.imshow(pixels)
    ax.set_xlim(0, pixels.width)
    ax.set_ylim(pixels.height, 0)
    ax.set_axis_off()
    ax.set_title(title, fontsize=14)

    for gt in image.objects:
        ax.add_patch(_rectangle(gt.box, GT_COLOR, linestyle="-", linewidth=2.5))

    for det in sorted(detections, key=lambda d: d.score):
        best_iou = max(
            (iou(det.box, gt.box) for gt in image.objects if gt.category_id == det.category_id),
            default=0.0,
        )
        above = best_iou >= iou_threshold
        color = MATCHED_COLOR if above else UNMATCHED_COLOR
        ax.add_patch(_rectangle(det.box, color, linestyle="-" if above else "--", linewidth=2))
        # Anchor the label at the box's bottom-left corner, kept inside the image.
        ax.text(
            min(max(det.box[0], 0), pixels.width * 0.85) + 3,
            min(max(det.box[1] + det.box[3], 20), pixels.height) - 5,
            f"IoU {best_iou:.2f} | score {det.score:.2f}",
            color="white",
            fontsize=8,
            clip_on=True,
            bbox={"facecolor": color, "edgecolor": "none", "pad": 1.5, "alpha": 0.85},
        )

    ax.legend(
        handles=[
            Line2D([], [], color=GT_COLOR, linewidth=2.5, label="Ground truth"),
            Line2D([], [], color=MATCHED_COLOR, linewidth=2, label=f"Prediction, IoU ≥ {iou_threshold}"),
            Line2D(
                [],
                [],
                color=UNMATCHED_COLOR,
                linewidth=2,
                linestyle="--",
                label=f"Prediction, IoU < {iou_threshold}",
            ),
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, 0),
        ncol=3,
        fontsize=10,
        frameon=False,
    )

    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(
            output,
            dpi=100,
            bbox_inches="tight",
            pil_kwargs={"quality": 85} if output.suffix == ".jpg" else None,
        )
    return fig


def _rectangle(box: Box, color: str, *, linestyle: str, linewidth: float) -> Rectangle:
    x, y, w, h = box
    return Rectangle((x, y), w, h, fill=False, edgecolor=color, linestyle=linestyle, linewidth=linewidth)


def plot_sweep(
    points: Sequence[SweepPoint],
    *,
    x_values: Sequence[float],
    x_label: str,
    title: str,
    subtitle: str = "",
    threshold: float | None = None,
    threshold_label: str = "",
    invert_x: bool = False,
    output: Path | None = None,
) -> Figure:
    """Plot every metric against the swept parameter.

    All four curves are oriented so that higher is better, which is why oLRP is shown as
    ``1 − oLRP``: the four lines are then directly comparable.
    """
    series = {
        "map50": [p.map50 for p in points],
        "map": [p.map for p in points],
        "lrp_quality": [1 - p.olrp if p.olrp is not None else 0.0 for p in points],
        "f1": [p.f1 for p in points],
    }

    fig, ax = plt.subplots(figsize=(8, 5))
    for key, values in series.items():
        label, color, linestyle = METRIC_STYLES[key]
        ax.plot(
            x_values,
            values,
            label=label,
            color=color,
            linestyle=linestyle,
            linewidth=2,
            marker="o",
            markersize=3.5,
        )

    if threshold is not None:
        ax.axvline(threshold, color="#64748b", linestyle=":", linewidth=1.5)
        ax.annotate(
            threshold_label,
            xy=(threshold, 0.55),
            xytext=(-8, 0),
            textcoords="offset points",
            rotation=90,
            va="center",
            ha="center",
            fontsize=9,
            color="#475569",
        )

    ax.set(xlabel=x_label, ylabel="Metric value (higher is better)", ylim=(-0.02, 1.05))
    if invert_x:
        ax.invert_xaxis()
    ax.set_title(subtitle, fontsize=9.5, color="#475569")
    fig.suptitle(title, fontsize=13, y=0.97)
    ax.grid(alpha=0.25)
    ax.legend(loc="best", fontsize=9, framealpha=0.9)

    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=130, bbox_inches="tight")
    return fig


TIDE_ERROR_COLORS = {
    "Cls": "#a855f7",
    "Loc": "#eab308",
    "Both": "#f97316",
    "Dupe": "#06b6d4",
    "Bkg": "#ef4444",
    "Miss": "#64748b",
}


def plot_tide_breakdown(
    points: Sequence[SweepPoint],
    *,
    x_values: Sequence[float],
    x_label: str,
    title: str,
    subtitle: str = "",
    threshold: float | None = None,
    threshold_label: str = "",
    invert_x: bool = False,
    output: Path | None = None,
) -> Figure:
    """Stack the AP that TIDE says each error type is responsible for, along a sweep.

    Where the metric plot only shows AP collapsing, this shows *what TIDE blames it on*
    at every point of the sweep.
    """
    if any(p.tide_errors is None for p in points):
        raise ValueError("Sweep points carry no TIDE breakdown; run the sweep with with_tide=True.")

    labels = list(TIDE_ERROR_COLORS)
    stacks = [[p.tide_errors.get(label, 0.0) for p in points] for label in labels]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.stackplot(
        x_values,
        *stacks,
        labels=labels,
        colors=[TIDE_ERROR_COLORS[label] for label in labels],
        alpha=0.9,
    )

    if threshold is not None:
        ax.axvline(threshold, color="#1e293b", linestyle=":", linewidth=1.5)
        ax.annotate(
            threshold_label,
            xy=(threshold, 0.55),
            xytext=(-8, 0),
            textcoords="offset points",
            rotation=90,
            va="center",
            ha="center",
            fontsize=9,
            color="#1e293b",
        )

    ax.set(xlabel=x_label, ylabel="AP recoverable by fixing this error type", ylim=(0, 1.05))
    if invert_x:
        ax.invert_xaxis()
    ax.set_title(subtitle, fontsize=9.5, color="#475569")
    fig.suptitle(title, fontsize=13, y=0.97)
    ax.grid(alpha=0.2)
    ax.legend(loc="upper left", fontsize=9, framealpha=0.9, title="TIDE error type")

    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=130, bbox_inches="tight")
    return fig
