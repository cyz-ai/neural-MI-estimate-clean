# `image` dataset package

This package builds image-pair benchmarks (Gaussian plots / rectangles) whose
ground-truth mutual information is fixed by construction, for evaluating neural
MI estimators.

## Acknowledgement

The image-generation code and the `utils/` helpers are **ported from** the paper:

> **Information Bottleneck Analysis of Deep Neural Networks via Lossy Compression**
> ICLR 2023

and its accompanying "Mutinfo" information-theoretic toolbox, distributed under
the GNU General Public License v3.0. Please cite that work if you use this code.

## Contents

Only the modules used by `datasets/image/image_dataset.py` are retained from the
upstream toolbox:

    utils/synthetic.py
    utils/dependent_norm.py

The upstream project's unused parts (the entropy / mutual-information
estimators, Keras/torch layers, example notebooks, gnuplot plotting scripts, and
packaging files) were removed. Refer to the upstream "Mutinfo" distribution for
the complete source and full GPL-3.0 text.
