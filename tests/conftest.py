"""Shared fixtures. Tests run offline on synthetic COCO-like annotations."""

from __future__ import annotations

import numpy as np
import pytest

from mapstudy.data import GroundTruth, ImageRecord


def make_images(n_images: int = 40, n_categories: int = 4, seed: int = 0) -> list[ImageRecord]:
    """Random images with 1-8 objects each, possibly overlapping, across a few categories."""
    rng = np.random.default_rng(seed)
    images = []
    for image_id in range(1, n_images + 1):
        width, height = 640, 480
        objects = []
        for _ in range(rng.integers(1, 9)):
            w, h = rng.uniform(10, 250, size=2)
            x, y = rng.uniform(0, width - w), rng.uniform(0, height - h)
            box = (float(x), float(y), float(w), float(h))
            objects.append(GroundTruth(image_id, int(rng.integers(n_categories)), box, float(w * h)))
        images.append(ImageRecord(image_id, width, height, tuple(objects)))
    return images


@pytest.fixture(scope="session")
def images() -> list[ImageRecord]:
    return make_images()


@pytest.fixture
def image_factory():
    return make_images
