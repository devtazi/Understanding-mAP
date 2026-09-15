"""Core data structures and MS-COCO 2017 loading.

Annotations are read from the ``HichTala/coco`` Parquet export of MS-COCO 2017 on the
Hugging Face Hub. Shards are downloaded once into the local Hugging Face cache and
read with :mod:`pyarrow`, which is much faster than the ``datasets`` streaming API and
avoids decoding images that the metric experiments do not need.
"""

from __future__ import annotations

import io
import json
from collections.abc import Iterator
from dataclasses import dataclass, field
from functools import lru_cache

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
from PIL import Image

from mapstudy.boxes import Box

DATASET_REPO = "HichTala/coco"
TRAIN_SHARDS = 39


@dataclass(frozen=True)
class GroundTruth:
    """An annotated object."""

    image_id: int
    category_id: int
    box: Box
    area: float


@dataclass(frozen=True)
class Detection:
    """A predicted object with its confidence score."""

    image_id: int
    category_id: int
    box: Box
    score: float


@dataclass(frozen=True)
class ImageRecord:
    """An image and its ground-truth objects. Pixels are only kept when requested."""

    image_id: int
    width: int
    height: int
    objects: tuple[GroundTruth, ...]
    image_bytes: bytes | None = field(default=None, repr=False)

    def load_image(self) -> Image.Image:
        if self.image_bytes is None:
            raise ValueError("Image pixels were not loaded; use include_pixels=True.")
        return Image.open(io.BytesIO(self.image_bytes)).convert("RGB")


def _shard_path(index: int) -> str:
    filename = f"data/train-{index:05d}-of-{TRAIN_SHARDS:05d}.parquet"
    return hf_hub_download(DATASET_REPO, filename, repo_type="dataset")


def _iter_rows(include_pixels: bool) -> Iterator[dict]:
    columns = ["image_id", "width", "height", "objects"]
    if include_pixels:
        columns.append("image")
    for shard in range(TRAIN_SHARDS):
        parquet = pq.ParquetFile(_shard_path(shard))
        for batch in parquet.iter_batches(batch_size=256, columns=columns):
            yield from batch.to_pylist()


def load_coco_images(n_images: int, *, include_pixels: bool = False) -> list[ImageRecord]:
    """Load the first ``n_images`` annotated images of the COCO 2017 train split.

    Images are taken in the dataset's stored order, so the subset is identical from
    one run to the next. Images without any annotated object are skipped.
    """
    if n_images <= 0:
        raise ValueError("n_images must be positive")

    records: list[ImageRecord] = []
    for row in _iter_rows(include_pixels):
        objects = row["objects"]
        if not objects["bbox"]:
            continue
        image_id = int(row["image_id"])
        gts = tuple(
            GroundTruth(image_id, int(category), tuple(float(v) for v in box), float(area))
            for box, category, area in zip(objects["bbox"], objects["category"], objects["area"], strict=True)
        )
        records.append(
            ImageRecord(
                image_id=image_id,
                width=int(row["width"]),
                height=int(row["height"]),
                objects=gts,
                image_bytes=row["image"]["bytes"] if include_pixels else None,
            )
        )
        if len(records) == n_images:
            break
    return records


@lru_cache(maxsize=1)
def category_names() -> tuple[str, ...]:
    """COCO category names, indexed by the ``category_id`` used in this dataset."""
    metadata = pq.ParquetFile(_shard_path(0)).schema_arrow.metadata[b"huggingface"]
    features = json.loads(metadata)["info"]["features"]
    return tuple(features["objects"]["category"]["feature"]["names"])
