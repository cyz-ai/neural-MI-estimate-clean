"""
Image-pair benchmarks with a target mutual information.

``image_dataset`` builds Gaussian-plot / rectangle image pairs whose ground-truth
MI is fixed by construction. Ported from "Information Bottleneck Analysis of Deep
Neural Networks via Lossy Compression" (ICLR 2024); its Mutinfo toolbox is
vendored under ``utils/`` (see ``README.md``).

Public helpers::

    from datasets.image import generate_gaussian_plot, visualize_images
"""

from .image_dataset import (
    generate_gaussian_plot,
    generate_rectangle_plot,
    visualize_images,
)

__all__ = [
    "generate_gaussian_plot",
    "generate_rectangle_plot",
    "visualize_images",
]
