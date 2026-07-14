<h1 align="center"> Mutual Information Estimation with Vector Copulas</h1>

<p align="center">
  <b>Vector Copula Estimator (VCE) — an MI estimator disentangling marginal patterns from dependence structure</b>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/NeurIPS-2025-8b5cf6.svg" alt="NeurIPS 2025">
  <img src="https://img.shields.io/badge/python-%E2%89%A53.9-blue.svg" alt="python">
  <img src="https://img.shields.io/badge/PyTorch-%E2%89%A52.0-ee4c2c.svg" alt="pytorch">
  <img src="https://img.shields.io/badge/CUDA-recommended-76b900.svg" alt="cuda">
</p>



1. **Marginal distribution learning.** Mapping the $X$ and $Y$ marginals onto $\mathcal{U}[0, 1]^{d_X}$ and $\mathcal{U}[0, 1]^{d_Y}$, stripping
   away their original patterns and shape.
2. **Vector copula (dependence) learning.** A mixture of vector Gaussian copulas is fit on the transformed data by maximum likelihood. MI can be directly read off from the copula.


---

## ✨ Highlights

- **Disentangles marginals and dependence** — isolating marginal effects from the
  dependence structure lets VCE model and learn the two parts separately and more flexibly.
- **Test-time search of the optimal copula** — train the copula mixture once, then select the
  dependence structure best suited to the data at test time (across different combination of copula
  components). No retraining needed.
- **6 estimators in one interface** — VCE plus MINE, InfoNCE, MRE, MINDE, and MIENF all share
  `learn(x, y)` / `MI(x, y)`.
- **7 benchmarks with exact ground-truth MI** — heavy-tailed, nonlinear, manifold, and image data.
- **Self-contained** — pure PyTorch with NumPy/SciPy; no external libraries required.

## 🚀 Installation

**Requirements:** Python ≥ 3.9, PyTorch ≥ 2.0, CUDA strongly recommended.

```bash
pip install -r requirements.txt
```

## ⚡ Quick Start

```python
import torch
from datasets import Spiral
from estimators import VCE

device = "cuda" if torch.cuda.is_available() else "cpu"

# A benchmark with closed-form MI. X, Y are each (n, dim//2); dim_x == dim_y.
dataset = Spiral(rho=0.7, dim=64, v=3.14 / 2)
X, Y = dataset.sample(n=10000)
X, Y = X.to(device).clone().detach(), Y.to(device).clone().detach()

class Hyperparams(object):
    def __init__(self):
        self.lr = 5e-4
        self.bs = 500
        self.wd = 1e-5
        self.max_iteration = 1250

estimator = VCE(Hyperparams())              
estimator.to(device)
estimator.learn(X, Y)
print("true MI:", dataset.MI())
print("est MI:", estimator.MI(X, Y))
```

The same setup runs in [`exp_spiral.ipynb`](exp_spiral.ipynb).

## 🗜️ High-Dimensional Data

Like any generative estimator, VCE degrades when the per-side dimension is very large. For
high-dimensional inputs such as **images** or **LLM embeddings**, we recommend compressing each side
to a low-dimensional latent first — with an **autoencoder** or **PCA** — and estimating MI on the latent. The
`compression/` module provides this:

```python
from compression import Autoencoder

k = 32                                            # target latent size (same for both sides)

# train a reconstruction, then encode each side independently
ae_x = Autoencoder(x_dim=X.size(1), latent_dim=k).to(device); ae_x.learn(X)
ae_y = Autoencoder(x_dim=Y.size(1), latent_dim=k).to(device); ae_y.learn(Y)
Zx, Zy = ae_x.encode(X), ae_y.encode(Y)           # each (n, k)

estimator.learn(Zx, Zy)                           # then estimate MI on the compressed codes
```

As long as the compression is near-lossless, the MI between the codes closely tracks the MI between
the original variables. You can check how lossless the code is with `ae_x.compressibility(X)`, which
returns a reconstruction score in ~[0, 1] (higher is better; ≈1 means near-lossless).

## 🧩 Estimators

Every estimator is an `nn.Module` exposing `learn(x, y)` (train) and `MI(x, y)` (read), and trains
through the shared `optimizer.py` (Adam, 80/20 split, early stopping).

| Estimator | Import | Family |
|---|---|---|
| **VCE** *(ours)* | `from estimators import VCE` | vector copula |
| MINE | `from estimators import MINE` | Donsker–Varadhan lower bound |
| InfoNCE | `from estimators import InfoNCE` | contrastive (CPC / NCE) bound |
| MRE | `from estimators import MRE` | MI via (reference-based) ratio estimation |
| MINDE | `from estimators import MINDE` | diffusion / score-based |
| MIENF | `from estimators import MIENF` | MI via pairs of normalizing-flow transformations |

Key VCE knobs (attributes on the `Hyperparams` object): `K_components` (copula mixture size,
default 32), `n_restarts` (best-of-*N* copula fits, default 4), `marginal_flow` (`"FM"` or `"NAF"`),
`bon_selection` (held-out component pruning, default `True`).

## 📊 Benchmarks

All ship with closed-form or exactly-computable ground-truth MI, adapted from [1, 2]. Each comes with a self-contained
notebook that samples the data, runs the estimators, and reports MI against ground truth — open and
run whichever you need.

| Benchmark | What it probes | Example |
|---|---|---|
| Wrapped Gaussian | nonlinear per-coordinate warps | [`exp_wrapped_Gaussian.ipynb`](exp_wrapped_Gaussian.ipynb) |
| Multivariate Student-t | heavy-tailed dependence | [`exp_synthetic_student_t.ipynb`](exp_synthetic_student_t.ipynb) |
| Mixture of Gaussians | multimodal block dependence | [`exp_synthetic_mog.ipynb`](exp_synthetic_mog.ipynb) |
| Smoothed uniform | bounded-support marginals | [`exp_smoothed_uniform.ipynb`](exp_smoothed_uniform.ipynb) |
| Swiss Roll | manifold-embedded copula | — |
| Spiral | norm-dependent rotation | [`exp_spiral.ipynb`](exp_spiral.ipynb) |
| Images with known MI | high-dimensional image pairs | [`exp_image_Gaussian_medium.ipynb`](exp_image_Gaussian_medium.ipynb) |
| Bert/Qwen IMDB embeddings via resampling | high-dimensional LM embeddings | TO DO |

[1] Czyż et al. [Beyond Normal: On the Evaluation of Mutual Information Estimators](https://arxiv.org/abs/2306.11078). NeurIPS 2023. *(synthetic benchmarks)*
[2] Butakov et al. [Information Bottleneck Analysis of Deep Neural Networks via Lossy Compression](https://arxiv.org/abs/2305.08013). ICLR 2024. *(image benchmark)*

## 📁 Project Structure

```
estimators/     MI estimators (VCE + 5 baselines), shared learn(x,y) / MI(x,y) interface
nde/            neural density estimators — flows (NAF, MAF, FM) and variational copulas (MoG, VGC)
datasets/       synthetic & image benchmarks with known ground-truth MI
compression/    autoencoder & PCA to compress high-dim inputs (images, embeddings) before MI
optimizer.py    shared training loop (Adam, train/val split, early stopping)
exp_*.ipynb     one self-contained experiment notebook per benchmark
```

## 📖 Citation

If you find the code and method useful, please consider citing the following paper:

```bibtex
@article{chen2025neural,
  title={Neural mutual information estimation with vector copulas},
  author={Chen, Yanzhi and Ou, Zijing and Weller, Adrian and Gutmann, Michael},
  journal={Advances in Neural Information Processing Systems},
  volume={38},
  pages={44803--44823},
  year={2025}
}
```
