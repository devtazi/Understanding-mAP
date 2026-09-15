"""Figures showing what each synthetic detector actually predicts."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

from mapstudy.boxes import Box, iou
from mapstudy.data import Detection, ImageRecord

GT_COLOR = "#22c55e"
MATCHED_COLOR = "#3b82f6"
UNMATCHED_COLOR = "#ef4444"


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
