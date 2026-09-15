"""Command-line interface.

mapstudy run --scenario a b            # evaluate scenarios A and B
mapstudy run --all --n-images 1000     # reproduce the results table
mapstudy visualize --scenario b        # draw one image with its predictions
"""

from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Sequence
from pathlib import Path

import matplotlib.pyplot as plt

from mapstudy.data import load_coco_images
from mapstudy.evaluation import EvaluationReport, evaluate
from mapstudy.reporting import format_results_table
from mapstudy.scenarios import SCENARIOS, generate_detections
from mapstudy.visualization import plot_detections

logger = logging.getLogger("mapstudy")

DEFAULT_SEED = 42
DEFAULT_N_IMAGES = 1000


def main(argv: Sequence[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(message)s")
    logging.getLogger("mapstudy").setLevel(logging.INFO)
    args.handler(args)


def run(args: argparse.Namespace) -> None:
    scenarios = list(SCENARIOS) if args.all else args.scenario
    logger.info("Loading %d annotated COCO 2017 images...", args.n_images)
    images = load_coco_images(args.n_images)

    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    reports: list[EvaluationReport] = []
    for name in scenarios:
        logger.info("Evaluating %s...", SCENARIOS[name].title)
        detections = generate_detections(images, name, seed=args.seed)
        report = evaluate(
            images,
            detections,
            scenario=name,
            seed=args.seed,
            tide_plot_dir=output_dir / "tide" if args.tide_plots else None,
        )
        (output_dir / f"{name}.json").write_text(json.dumps(report.to_dict(), indent=2))
        reports.append(report)

    table = format_results_table(reports)
    (output_dir / "summary.md").write_text(table + "\n")
    logger.info("\n%s\n\nResults written to %s/", table, output_dir)


def visualize(args: argparse.Namespace) -> None:
    images = load_coco_images(args.image_index + 1, include_pixels=True)
    image = images[args.image_index]
    scenario = SCENARIOS[args.scenario]
    detections = generate_detections([image], args.scenario, seed=args.seed)

    output = args.output or Path("docs/figures") / f"scenario_{args.scenario}.jpg"
    fig = plot_detections(image, detections, title=scenario.title, output=output)
    plt.close(fig)
    logger.info("Figure written to %s", output)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mapstudy",
        description="Controlled experiments on the blind spots of mean Average Precision.",
    )
    subparsers = parser.add_subparsers(required=True)

    run_parser = subparsers.add_parser("run", help="Evaluate synthetic detectors with every metric.")
    selection = run_parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--scenario", nargs="+", type=str.lower, choices=SCENARIOS)
    selection.add_argument("--all", action="store_true", help="Run every scenario.")
    run_parser.add_argument("--n-images", type=_positive_int, default=DEFAULT_N_IMAGES)
    run_parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    run_parser.add_argument("--output-dir", type=Path, default=Path("results"))
    run_parser.add_argument(
        "--no-tide-plots", dest="tide_plots", action="store_false", help="Skip TIDE figures."
    )
    run_parser.set_defaults(handler=run)

    viz_parser = subparsers.add_parser("visualize", help="Draw a scenario's predictions on an image.")
    viz_parser.add_argument("--scenario", type=str.lower, choices=SCENARIOS, required=True)
    viz_parser.add_argument("--image-index", type=int, default=0, help="Index among annotated images.")
    viz_parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    viz_parser.add_argument("--output", type=Path, help="Defaults to docs/figures/scenario_<name>.jpg")
    viz_parser.set_defaults(handler=visualize)

    return parser


def _positive_int(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number
