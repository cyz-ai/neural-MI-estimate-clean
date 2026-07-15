#!/usr/bin/env python3
"""run_bench.py — single-file MI-estimator benchmark: run cells across GPUs + draw the 1x4 figure.

Grid: {estimators} x {datasets} x {dims} x 2 seeds. Each cell is ONE process that writes one JSON
(both seeds inside), dispatched over a pool of GPUs (slowest/highest-dim first, resumable). `plot`
renders the 1x4 accuracy figure — I(X;Y) vs d, one panel per dataset, ground-truth curve dashed —
from those JSONs.

Config (uniform, for fairness): n=10000, rho=0.7, lr=5e-4, bs=500, wd=1e-5, max_iteration=1250.
A 750s wall-clock cap per training run is enforced by optimizer.learn(timeout=...) (best-so-far kept).

Actions (composable)
  --exp    orchestrate the grid across idle GPUs (resumable; --force to overwrite existing JSONs)
  --plot   draw results/benchmark_scan_all_dim.png (1x4) + results/benchmark_scan_time.md  [+ --panels]
  --cell E DATASET DIM   run ONE cell, both seeds  [worker; used internally by --exp]

Examples
  nohup python run_bench.py --exp --estimators MINDE,InfoNCE --gpus 1,2,7 > scan.log 2>&1 &
  python run_bench.py --exp --plot                  # run the grid, then draw figure + time table
  python run_bench.py --plot                        # figure + time table only, from existing JSONs
  CUDA_VISIBLE_DEVICES=4 python run_bench.py --cell VCE spiral 64
"""
import os
import sys
import json
import time
import argparse
import warnings

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)                 # repo root -> import optimizer / estimators / datasets
warnings.filterwarnings("ignore")

# ----------------------------------------------------------------------------------- config
RESULT_DIR = os.path.join(HERE, 'results', 'benchmark_scan')
FIG_DIR    = os.path.join(HERE, 'results')          # figure + time table land directly in results/
LATENT_DIR = os.path.join(RESULT_DIR, 'latents')   # VCE flow latents (seed 0) for later MoG refits
LOG_DIR    = os.path.join(RESULT_DIR, 'logs')

N, RHO     = 10000, 0.7
MAX_ITER   = 1250            # uniform training-iteration budget (the wall-clock cap lives in optimizer.learn)
T_PATIENCE = 500
LR, BS, WD = 5e-4, 500, 1e-5
SEEDS      = (0, 1)          # "two runs each"
DEVICE     = 'cuda:0'        # CUDA_VISIBLE_DEVICES masks the chosen GPU to cuda:0

ESTIMATORS   = ['MINE', 'InfoNCE', 'MINDE', 'MIENF', 'VCE']
DATASETS     = ['wrapped', 'mog', 'student_t', 'spiral']
DIMS         = [8, 16, 32, 64, 128]
DEFAULT_GPUS = [1, 2, 7]

# plot style: marker spec = matplotlib color char + marker char; plus legend label
STYLE = {
    'MINE':    ('rd', 'MINE'),
    'InfoNCE': ('bs', 'InfoNCE'),
    'MINDE':   ('g*', 'MINDE'),
    'MIENF':   ('yh', r'$\mathcal{N}$-MIENF'),
    'VCE':     ('mo', 'VCE'),
}
TITLES = {'wrapped': 'Wrapped Gaussian', 'mog': 'MoG',
          'student_t': 'Student-t (v=1)', 'spiral': 'Spiral'}
CLIP_LOW, STD_CAP = -1.0, 2.25     # clip low blow-ups (MINE heavy tails); cap error bars for readability


def _cell_json(estimator, dataset, dim):
    return os.path.join(RESULT_DIR, f'{estimator}_{dataset}_{dim}.json')


# ============================================================ experiment: data + estimators
def set_seed(s):
    import numpy as np
    import torch
    np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def make_data(dataset, d):
    """Return (X, Y, gt) for one dataset at total-dim d (each side d//2), matching the notebooks."""
    import numpy as np
    import torch
    if dataset == 'wrapped':
        from datasets import NonlinearGaussian
        ds = NonlinearGaussian(n_samples=N, n_dims=d, rho=RHO, mu=0, case='3a')
        X0, Y0 = ds.sample_data(n_samples=N)
        X, Y = ds.transformation(X0, Y0)
        gt = ds.true_mutual_info()
    elif dataset == 'mog':
        from datasets import MoG
        ds = MoG(n_samples=N, n_dims=d, K=5,
                 shifts=[-0.2, -0.1, 0, 0.3, 0.4], rhos=[-0.3, 0.5, 0.2, 0.4, 0.9])
        X, Y = ds.sample_data(n_samples=N)
        gt = ds.empirical_mutual_info()
    elif dataset == 'student_t':
        from datasets import MultivariateStudentT
        dx = dy = d // 2
        disp = np.eye(dx + dy)
        disp[0:dx, dx:] = np.eye(dx) * RHO
        disp[dx:, 0:dx] = np.eye(dx) * RHO
        ds = MultivariateStudentT(dim_x=dx, dim_y=dy, mean=np.zeros(dx + dy), dispersion=disp, df=1)
        Xn, Yn = ds.sample(N)
        X, Y = torch.Tensor(Xn).float(), torch.Tensor(Yn).float()
        gt = float(ds.mutual_information())
    elif dataset == 'spiral':
        from datasets import Spiral
        ds = Spiral(rho=RHO, dim=d, v=3.14 / 2)
        X, Y = ds.sample(n=N)
        gt = ds.MI()
    else:
        raise ValueError(f'unknown dataset: {dataset}')
    X = X.to(DEVICE).clone().detach().float()
    Y = Y.to(DEVICE).clone().detach().float()
    return X, Y, float(gt)


def make_hp(d):
    """Shared Hyperparams object (a plain namespace, matching the estimators' duck-typed reads)."""
    class HP:
        pass
    hp = HP()
    hp.lr, hp.bs, hp.wd = LR, BS, WD
    hp.max_iteration = MAX_ITER
    hp.t_patience = T_PATIENCE
    hp.dim = d // 2
    hp.device = DEVICE
    hp.importance_sampling = True          # MINDE
    return hp


def make_estimator(name, d, hp):
    arch = [d, 500, 500, 500, 1]           # concat-[x,y] critic MLP for the f-divergence methods
    if name == 'MINE':
        from estimators import MINE;    return MINE(arch, hp)
    if name == 'InfoNCE':
        from estimators import InfoNCE; return InfoNCE(arch, hp)
    if name == 'MINDE':
        from estimators import MINDE;   return MINDE(hp)
    if name == 'MIENF':
        from estimators import MIENF;   return MIENF(hp)
    if name == 'VCE':
        from estimators import VCE;     return VCE(hp)
    raise ValueError(f'unknown estimator: {name}')


def _maybe_cache_vce_latents(est, estimator, dataset, dim, seed, gt):
    """Persist VCE flow latents (seed 0) so the MoG copula can be refit later without re-flowing."""
    import torch
    if estimator == 'VCE' and seed == 0 and getattr(est, '_cached_latents', None) is not None:
        os.makedirs(LATENT_DIR, exist_ok=True)
        v, w = est._cached_latents
        torch.save({'v': v.detach().cpu(), 'w': w.detach().cpu(), 'gt': gt,
                    'benchmark': dataset, 'dim': dim},
                   os.path.join(LATENT_DIR, f'VCE_{dataset}_{dim}_seed0.pt'))


def run_cell(estimator, dataset, dim):
    """Run one (estimator, dataset, dim) cell over all seeds; write one JSON. Returns its path."""
    import torch  # noqa: F401  (ensures CUDA context / clear error if torch missing)
    os.makedirs(RESULT_DIR, exist_ok=True)
    runs = []
    for seed in SEEDS:
        rec = {'seed': seed}
        try:
            set_seed(seed)
            X, Y, gt = make_data(dataset, dim)
            est = make_estimator(estimator, dim, make_hp(dim)).to(DEVICE)
            t0 = time.time()
            est.learn(X, Y)
            train_s = time.time() - t0
            mi = float(est.MI(X, Y))
            rec.update(gt=gt, est=mi, abs_err=abs(mi - gt), time_s=round(train_s, 2))
            _maybe_cache_vce_latents(est, estimator, dataset, dim, seed, gt)
            print(f'[{estimator}/{dataset}/d={dim}/seed{seed}] est={mi:.4f} gt={gt:.4f} '
                  f'|err|={abs(mi - gt):.4f} {train_s:.1f}s', flush=True)
        except Exception as e:
            import traceback
            rec.update(error=f'{type(e).__name__}: {e}')
            print(f'[{estimator}/{dataset}/d={dim}/seed{seed}] ERROR {e}', flush=True)
            traceback.print_exc()
        runs.append(rec)

    out = {'estimator': estimator, 'benchmark': dataset, 'dim': dim, 'runs': runs}
    path = _cell_json(estimator, dataset, dim)
    with open(path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f'wrote {path}', flush=True)
    return path


# ==================================================================== orchestrate across GPUs
def run_scan(estimators, datasets, dims, gpus, force=False):
    """Dispatch (estimator x dataset x dim) cells across `gpus`, highest-dim first, resumable."""
    import subprocess
    os.makedirs(RESULT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    jobs = [(e, b, d) for e in estimators for b in datasets for d in dims]
    jobs.sort(key=lambda j: j[2], reverse=True)                     # front-load slow (high-dim) cells
    todo = [j for j in jobs if force or not os.path.exists(_cell_json(*j))]
    print(f'scan: {len(jobs)} cells, {len(jobs) - len(todo)} already done, '
          f'{len(todo)} to run on GPUs {gpus}', flush=True)

    running, done, t_start = {}, 0, time.time()                    # gpu -> (proc, job, logfile, start)
    while todo or running:
        for gpu in gpus:                                           # fill idle GPUs
            if gpu not in running and todo:
                e, b, d = todo.pop(0)
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
                lf = open(os.path.join(LOG_DIR, f'{e}_{b}_{d}.log'), 'w')
                p = subprocess.Popen(
                    [sys.executable, os.path.abspath(__file__), '--cell', e, b, str(d)],
                    env=env, stdout=lf, stderr=subprocess.STDOUT)
                running[gpu] = (p, (e, b, d), lf, time.time())
                print(f'[+] GPU{gpu} <- {e}/{b}/d={d}   ({len(todo)} queued)', flush=True)
        for gpu in list(running):                                  # reap finished
            p, job, lf, start = running[gpu]
            if p.poll() is not None:
                lf.close(); done += 1; e, b, d = job
                print(f'[-] GPU{gpu} done {e}/{b}/d={d} rc={p.returncode} ({time.time() - start:.0f}s) | '
                      f'{done} done, {len(todo)} queued, {(time.time() - t_start) / 60:.1f} min', flush=True)
                del running[gpu]
        time.sleep(5)
    print(f'SCAN COMPLETE: {done} cells in {(time.time() - t_start) / 60:.1f} min', flush=True)


# ============================================================ results -> curves -> figures
def load_cell(estimator, dataset, dim):
    """(mean_est, std_est, mean_gt) over seeds.

    A cell with ANY missing / NaN / non-finite estimate (a crash or numerical blow-up, e.g.
    MIENF on heavy-tailed student-t) is treated as failed and reads est=0 — matching the
    notebook logic. A cell is averaged only when every seed produced a finite estimate.
    """
    import numpy as np
    fn = _cell_json(estimator, dataset, dim)
    if not os.path.isfile(fn):
        return 0.0, 0.0, np.nan
    runs = json.load(open(fn)).get('runs', [])
    ests = [r.get('est') for r in runs]
    gts = [r['gt'] for r in runs if r.get('gt') is not None]
    gt = float(np.mean(gts)) if gts else np.nan
    if not ests or any(e is None or not np.isfinite(e) for e in ests):
        return 0.0, 0.0, gt                         # any bad seed -> whole cell reads 0
    est = np.array(ests, dtype=float)
    return float(est.mean()), float(est.std()), gt


def build_curves(estimators, datasets, dims):
    """Return mi[ds][e], sd[ds][e] (lists over dims) and gt[ds] (mean truth per dim)."""
    import numpy as np
    mi, sd, gt = {}, {}, {}
    for ds in datasets:
        mi[ds] = {e: [] for e in estimators}
        sd[ds] = {e: [] for e in estimators}
        gt_curve = []
        for dim in dims:
            gts_here = []
            for e in estimators:
                m, s, g = load_cell(e, ds, dim)
                m = max(m, CLIP_LOW)
                s = min(max(s, 0.03 * abs(m)), STD_CAP)
                mi[ds][e].append(m)
                sd[ds][e].append(s)
                if np.isfinite(g):
                    gts_here.append(g)
            gt_curve.append(np.mean(gts_here) if gts_here else np.nan)
        gt[ds] = gt_curve
    return mi, sd, gt


def _pyplot():
    import matplotlib
    matplotlib.use('Agg')                           # headless / file-only rendering
    import matplotlib.pyplot as plt
    return plt


def _draw_accuracy_panel(ax, ds, dims, estimators, mi, sd, gt, legend=False):
    """One dataset panel: I(X;Y) vs d for each estimator + the dashed ground-truth curve."""
    import numpy as np
    xs = list(range(len(dims)))
    top = np.nanmax(gt[ds]) * 1.25
    ax.plot(xs, gt[ds], color='k', ls='--', lw=1.5, marker='^', ms=7, mfc='none', label=r'$I(X; Y)$')
    for e in estimators:
        color, mk = STYLE[e][0][:-1], STYLE[e][0][-1]
        ax.plot(xs, mi[ds][e], marker=mk, color=color, ms=8, lw=1, ls='dotted', mfc='none', label=STYLE[e][1])
        ax.errorbar(xs, mi[ds][e], sd[ds][e], fmt='none', capsize=1, ecolor=color)
    ax.set_ylim(CLIP_LOW - 0.5, top)                # axis keyed to truth; high outliers run off-top
    ax.set_xticks(xs); ax.set_xticklabels(dims)
    ax.set_xlabel(r'$d$'); ax.set_ylabel(r'$I(X; Y)$'); ax.set_title(TITLES.get(ds, ds))
    if legend:
        ax.legend(fontsize=9)


def plot_accuracy_grid(estimators, datasets, dims, save=True):
    """The headline 1x4 figure (one panel per dataset). Returns the Figure."""
    import numpy as np
    plt = _pyplot()
    mi, sd, gt = build_curves(estimators, datasets, dims)
    plt.rcParams.update({'font.size': 12})
    fig, axes = plt.subplots(1, len(datasets), figsize=(5 * len(datasets), 5), dpi=90)
    axes = np.atleast_1d(axes)
    for j, (ds, ax) in enumerate(zip(datasets, axes)):
        _draw_accuracy_panel(ax, ds, dims, estimators, mi, sd, gt, legend=(j == 0))
    fig.tight_layout()
    if save:
        os.makedirs(FIG_DIR, exist_ok=True)
        out = os.path.join(FIG_DIR, 'benchmark_scan_all_dim.png')
        fig.savefig(out, dpi=400, bbox_inches='tight')
        print(f'wrote {out}')
    return fig


def plot_accuracy_panels(estimators, datasets, dims, save=True):
    """Per-dataset individual figures (same panels as the grid, one PNG each)."""
    plt = _pyplot()
    mi, sd, gt = build_curves(estimators, datasets, dims)
    plt.rcParams.update({'font.size': 12})
    for j, ds in enumerate(datasets):
        fig, ax = plt.subplots(figsize=(5, 5), dpi=80)
        _draw_accuracy_panel(ax, ds, dims, estimators, mi, sd, gt,
                             legend=(j == 0 or j == len(datasets) - 1))
        fig.tight_layout()
        if save:
            os.makedirs(FIG_DIR, exist_ok=True)
            out = os.path.join(FIG_DIR, f'benchmark_scan_{ds}_dim.png')
            fig.savefig(out, dpi=400, bbox_inches='tight')
            print(f'wrote {out}')
        plt.close(fig)


def report_time(estimators, datasets, dims, save=True):
    """Mean training time per estimator (over all cells/seeds) as a 2-row markdown table.

    Row 1 = method names, row 2 = mean exec time (s); one column per estimator. Prints the
    table and (save=True) writes it to results/figures/benchmark_scan_time.md.
    """
    import numpy as np
    means = {}
    for e in estimators:
        ts = []
        for ds in datasets:
            for d in dims:
                fn = _cell_json(e, ds, d)
                if not os.path.isfile(fn):
                    continue
                for r in json.load(open(fn)).get('runs', []):
                    t = r.get('time_s')
                    if t is not None and np.isfinite(t):
                        ts.append(t)
        means[e] = float(np.mean(ts)) if ts else float('nan')

    header = '| ' + ' | '.join(estimators) + ' |'
    sep    = '| ' + ' | '.join('---' for _ in estimators) + ' |'
    row    = '| ' + ' | '.join(f'{means[e]:.0f}s' if means[e] == means[e] else '—' for e in estimators) + ' |'
    table  = '### Mean training time per estimator (seconds)\n\n' + '\n'.join([header, sep, row]) + '\n'
    print(table)
    if save:
        os.makedirs(FIG_DIR, exist_ok=True)
        out = os.path.join(FIG_DIR, 'benchmark_scan_time.md')
        with open(out, 'w') as f:
            f.write(table)
        print(f'wrote {out}')
    return table


# ================================================================================== CLI
def _csv(s):
    return [x for x in s.split(',') if x]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    # actions (composable): --exp runs the grid, --plot draws figures, both -> run then plot
    ap.add_argument('--exp', action='store_true', help='run the experiment grid across GPUs')
    ap.add_argument('--plot', action='store_true', help='draw the 1x4 accuracy figure (+ --panels, --time)')
    ap.add_argument('--cell', nargs=3, metavar=('EST', 'DATASET', 'DIM'),
                    help='worker: run ONE cell, both seeds (used internally by --exp)')
    # grid / knobs
    ap.add_argument('--estimators', default=','.join(ESTIMATORS))
    ap.add_argument('--datasets', default=','.join(DATASETS))
    ap.add_argument('--dims', default=','.join(map(str, DIMS)))
    ap.add_argument('--gpus', default=','.join(map(str, DEFAULT_GPUS)))
    ap.add_argument('--force', action='store_true', help='re-run cells even if their JSON exists (--exp)')
    ap.add_argument('--panels', action='store_true', help='also save per-dataset panels (--plot)')
    a = ap.parse_args()

    if a.cell:                                          # worker mode: one cell, then exit
        run_cell(a.cell[0], a.cell[1], int(a.cell[2]))
        return
    if not (a.exp or a.plot):
        ap.error('nothing to do: pass --exp and/or --plot (or --cell for a single worker cell)')

    ests, dss, dms = _csv(a.estimators), _csv(a.datasets), [int(x) for x in _csv(a.dims)]
    if a.exp:
        run_scan(ests, dss, dms, [int(x) for x in _csv(a.gpus)], a.force)
    if a.plot:
        plot_accuracy_grid(ests, dss, dms)
        report_time(ests, dss, dms)                    # time table always accompanies --plot
        if a.panels:
            plot_accuracy_panels(ests, dss, dms)


if __name__ == '__main__':
    main()
