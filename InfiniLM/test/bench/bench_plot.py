"""
Visualization layer for the end-to-end Fused-FFN benchmark.

This module renders a suite of charts from the per-scenario latency
samples produced by ``bench_fused_ffn.py``. Every chart is derived from
end-to-end wall-clock latency only — there is no per-layer FFN data
anywhere in this pipeline, matching the benchmark's controlled-variable
design.

Charts produced
---------------
    01_e2e_speedup_overview.<fmt>  — one horizontal bar per scenario
    02_e2e_nf_vs_f.<fmt>           — grouped bars, NF vs F mean latency
                                     with stdev error bars
    03_speedup_heatmap.<fmt>       — (batch_size × input_tokens) heatmap
    04_latency_distribution.<fmt>  — box plots over measured rounds
    05_throughput_scaling.<fmt>    — batch / sequence scaling curves
    06_paired_delta.<fmt>          — per-round latency delta (F − NF),
                                     only when order="interleaved"
    07_correctness.<fmt>           — correctness info card

Usage from bench_fused_ffn.py::

    from bench_plot import save_e2e_plots
    save_e2e_plots(args, verify_data, all_samples, TEST_CASES,
                   out_dir=args.plot, fmt=args.plot_format)

``matplotlib`` is imported lazily so the bench script still works
without it when ``--plot`` is not requested.
"""

from __future__ import annotations

import sys
from pathlib import Path
from statistics import mean, stdev


# ── matplotlib lazy loader ──────────────────────────────────────────


def _require_matplotlib():
    """Import matplotlib in headless (Agg) mode."""
    try:
        import matplotlib
        if "matplotlib.pyplot" not in sys.modules:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # noqa: F401
        return matplotlib
    except ImportError as e:
        raise RuntimeError(
            "matplotlib is required for --plot. "
            "Install it with: pip install matplotlib"
        ) from e


# ── shared helpers ──────────────────────────────────────────────────


def _build_meta(args):
    return {
        "device":   args.device,
        "n_device": args.n_device,
        "model":    Path(args.model_path).name,
        "rounds":   args.rounds,
        "warmup":   args.warmup,
        "order":    getattr(args, "order", "interleaved"),
    }


def _title_suffix(meta: dict) -> str:
    return (f"{meta['device']}×{meta['n_device']}  ·  "
            f"rounds={meta['rounds']} ({meta['order']})  ·  "
            f"{meta['model']}")


def _speedup_pct(nf: float, f: float) -> float:
    return ((nf - f) / nf * 100.0) if nf and nf > 0 else 0.0


def _is_seq1(label: str) -> bool:
    return "seq=1" in label


def _savefig(fig, path: Path, fmt: str):
    import matplotlib.pyplot as plt
    path = Path(path).with_suffix(f".{fmt}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def _nf_f_mean_stdev(samples: dict):
    """Return (nf_mean, f_mean, nf_sd, f_sd) from ``{'nf': [...], 'f': [...]}``."""
    nf = samples["nf"]
    f = samples["f"]
    nf_m = mean(nf) if nf else 0.0
    f_m = mean(f) if f else 0.0
    nf_sd = stdev(nf) if len(nf) > 1 else 0.0
    f_sd = stdev(f) if len(f) > 1 else 0.0
    return nf_m, f_m, nf_sd, f_sd


# ── Chart 1: speedup overview ───────────────────────────────────────


def _plot_speedup_overview(all_samples, test_cases, out, meta, fmt):
    _require_matplotlib()
    import matplotlib.pyplot as plt

    labels = [c["label"] for c in test_cases]
    speedups = []
    for label in labels:
        nf_m, f_m, _, _ = _nf_f_mean_stdev(all_samples[label])
        speedups.append(_speedup_pct(nf_m, f_m))

    colors = ["#2ecc71" if s >= 0 else "#e74c3c" for s in speedups]
    y = list(range(len(labels)))

    fig, ax = plt.subplots(figsize=(10, max(4, 0.38 * len(labels))))
    bars = ax.barh(y, speedups, color=colors, edgecolor="#333", height=0.7)

    for i, label in enumerate(labels):
        if _is_seq1(label):
            ax.axhspan(i - 0.5, i + 0.5, alpha=0.07,
                       color="#3498db", zorder=-1)

    ax.axvline(0, color="black", lw=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("E2E Speedup (%)   ←  negative = slower")
    ax.set_title(f"Fused FFN End-to-End Speedup\n{_title_suffix(meta)}",
                 fontsize=11)
    ax.grid(axis="x", linestyle="--", alpha=0.4)

    smax = max((abs(s) for s in speedups), default=1.0)
    smax = max(smax, 1.0)
    ax.set_xlim(-smax * 1.2, smax * 1.3)

    for i, (bar, s) in enumerate(zip(bars, speedups)):
        x = bar.get_width()
        ha = "left" if x >= 0 else "right"
        off = smax * 0.02 * (1 if x >= 0 else -1)
        ax.text(x + off, i, f"{s:+.1f}%",
                va="center", ha=ha, fontsize=8)

    _savefig(fig, out / "01_e2e_speedup_overview", fmt)


# ── Chart 2: NF vs F mean latency (grouped bars with stdev bars) ────


def _plot_nf_vs_f(all_samples, test_cases, out, meta, fmt):
    _require_matplotlib()
    import matplotlib.pyplot as plt
    import numpy as np

    labels = [c["label"] for c in test_cases]
    nf_m, f_m, nf_sd, f_sd = [], [], [], []
    for label in labels:
        a, b, c, d = _nf_f_mean_stdev(all_samples[label])
        nf_m.append(a); f_m.append(b); nf_sd.append(c); f_sd.append(d)
    nf_m = np.asarray(nf_m); f_m = np.asarray(f_m)
    nf_sd = np.asarray(nf_sd); f_sd = np.asarray(f_sd)

    y = np.arange(len(labels))
    w = 0.38

    fig, ax = plt.subplots(figsize=(10, max(5, 0.42 * len(labels))))
    ax.barh(y - w / 2, nf_m, w, xerr=nf_sd,
            label="Non-Fused", color="#95a5a6", edgecolor="#333",
            error_kw={"lw": 0.8, "ecolor": "#444"})
    ax.barh(y + w / 2, f_m, w, xerr=f_sd,
            label="Fused", color="#3498db", edgecolor="#333",
            error_kw={"lw": 0.8, "ecolor": "#144a72"})
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("End-to-End Latency (ms, mean ± stdev)")
    ax.set_title(f"Non-Fused vs Fused End-to-End Latency\n{_title_suffix(meta)}",
                 fontsize=11)
    ax.grid(axis="x", linestyle="--", alpha=0.4)
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    _savefig(fig, out / "02_e2e_nf_vs_f", fmt)


# ── Chart 3: speedup heatmap (bs × input_tokens) ────────────────────


def _plot_speedup_heatmap(all_samples, test_cases, out, meta, fmt):
    _require_matplotlib()
    import matplotlib.pyplot as plt
    import numpy as np

    bs_vals = sorted({c["batch_size"] for c in test_cases})
    nin_vals = sorted({c["input_tokens"] for c in test_cases})
    bi = {b: i for i, b in enumerate(bs_vals)}
    ni = {n: j for j, n in enumerate(nin_vals)}

    grid = np.full((len(bs_vals), len(nin_vals)), np.nan)
    for c in test_cases:
        label = c["label"]
        nf_m, f_m, _, _ = _nf_f_mean_stdev(all_samples[label])
        grid[bi[c["batch_size"]], ni[c["input_tokens"]]] = _speedup_pct(nf_m, f_m)

    finite = grid[~np.isnan(grid)]
    vmax = float(np.max(np.abs(finite))) if finite.size else 1.0
    vmax = max(vmax, 1.0)

    fig, ax = plt.subplots(
        figsize=(1.4 + 0.9 * len(nin_vals), 1.2 + 0.6 * len(bs_vals))
    )
    cmap = plt.get_cmap("RdYlGn").copy()
    cmap.set_bad(color="#eeeeee")

    im = ax.imshow(grid, cmap=cmap, vmin=-vmax, vmax=vmax, aspect="auto")

    ax.set_xticks(range(len(nin_vals)))
    ax.set_xticklabels(nin_vals)
    ax.set_yticks(range(len(bs_vals)))
    ax.set_yticklabels(bs_vals)
    ax.set_xlabel("input_tokens (seq)")
    ax.set_ylabel("batch_size")
    ax.set_title(f"E2E Speedup Heatmap (%)\n{_title_suffix(meta)}",
                 fontsize=11)

    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            val = grid[i, j]
            if np.isnan(val):
                continue
            txtcolor = "black" if abs(val) < vmax * 0.55 else "white"
            ax.text(j, i, f"{val:+.1f}", ha="center", va="center",
                    color=txtcolor, fontsize=9)

    cbar = fig.colorbar(im, ax=ax, shrink=0.85)
    cbar.set_label("E2E Speedup (%)")
    _savefig(fig, out / "03_speedup_heatmap", fmt)


# ── Chart 4: latency distribution across rounds ────────────────────


def _plot_latency_distribution(all_samples, test_cases, out, meta, fmt):
    _require_matplotlib()
    import matplotlib.pyplot as plt

    labels = [c["label"] for c in test_cases]
    nf_data = [all_samples[l]["nf"] for l in labels]
    f_data = [all_samples[l]["f"] for l in labels]

    keep = [i for i, (a, b) in enumerate(zip(nf_data, f_data)) if a and b]
    if not keep:
        return
    labels = [labels[i] for i in keep]
    nf_data = [nf_data[i] for i in keep]
    f_data = [f_data[i] for i in keep]

    fig, ax = plt.subplots(figsize=(max(8, 0.65 * len(labels)), 6))
    positions = list(range(len(labels)))
    w = 0.35

    bp_nf = ax.boxplot(
        nf_data, positions=[p - w / 2 for p in positions],
        widths=w, patch_artist=True,
        boxprops=dict(facecolor="#bdc3c7", edgecolor="#333"),
        medianprops=dict(color="black"),
        whiskerprops=dict(color="#333"),
        capprops=dict(color="#333"),
        flierprops=dict(marker="o", markersize=3,
                        markerfacecolor="#7f8c8d", markeredgecolor="none"),
    )
    bp_f = ax.boxplot(
        f_data, positions=[p + w / 2 for p in positions],
        widths=w, patch_artist=True,
        boxprops=dict(facecolor="#3498db", edgecolor="#144a72"),
        medianprops=dict(color="black"),
        whiskerprops=dict(color="#144a72"),
        capprops=dict(color="#144a72"),
        flierprops=dict(marker="o", markersize=3,
                        markerfacecolor="#2980b9", markeredgecolor="none"),
    )

    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("E2E latency (ms)")
    ax.set_title(f"E2E Latency Distribution Across Rounds\n{_title_suffix(meta)}",
                 fontsize=11)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.legend([bp_nf["boxes"][0], bp_f["boxes"][0]],
              ["Non-Fused", "Fused"], loc="upper left", fontsize=9)
    fig.tight_layout()
    _savefig(fig, out / "04_latency_distribution", fmt)


# ── Chart 5: throughput scaling curves ─────────────────────────────


def _plot_throughput_scaling(all_samples, test_cases, out, meta, fmt):
    _require_matplotlib()
    import matplotlib.pyplot as plt

    def thr(case, key):
        label = case["label"]
        nf_m, f_m, _, _ = _nf_f_mean_stdev(all_samples[label])
        total_tokens = case["batch_size"] * case["input_tokens"]
        if key == "nf":
            return total_tokens / (nf_m / 1000.0) if nf_m > 0 else 0.0
        return total_tokens / (f_m / 1000.0) if f_m > 0 else 0.0

    batch_cases = sorted(
        [c for c in test_cases if c["input_tokens"] == 1],
        key=lambda c: c["batch_size"],
    )
    seq_cases = sorted(
        [c for c in test_cases if c["batch_size"] == 1],
        key=lambda c: c["input_tokens"],
    )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    def _draw(ax, cases, xkey, xlabel, logx):
        if not cases:
            ax.set_visible(False)
            return
        xs = [c[xkey] for c in cases]
        nf = [thr(c, "nf") for c in cases]
        f = [thr(c, "f") for c in cases]
        ax.plot(xs, nf, "o-", color="#7f8c8d", lw=1.8, ms=6, label="Non-Fused")
        ax.plot(xs, f, "o-", color="#2980b9", lw=1.8, ms=6, label="Fused")
        if logx:
            ax.set_xscale("log", base=2)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Throughput (tokens/s)")
        ax.grid(linestyle="--", alpha=0.4)
        ax.legend(fontsize=8)

    _draw(ax1, batch_cases, "batch_size",
          "batch_size  (input_tokens = 1)", logx=False)
    ax1.set_title("Batch Scaling (seq=1, GEMV)")
    _draw(ax2, seq_cases, "input_tokens",
          "input_tokens  (batch_size = 1, log₂)", logx=True)
    ax2.set_title("Sequence Scaling (bs=1, GEMM)")

    fig.suptitle(f"End-to-End Throughput Scaling\n{_title_suffix(meta)}",
                 fontsize=11)
    fig.tight_layout()
    _savefig(fig, out / "05_throughput_scaling", fmt)


# ── Chart 6: per-round paired delta (interleaved only) ─────────────


def _plot_paired_delta(all_samples, test_cases, out, meta, fmt):
    """
    Under interleaved ordering, sample r of NF and sample r of F were
    taken back-to-back. Plotting (f_r - nf_r) per round exposes whether
    the delta is stable over time or drifting due to thermal/clock
    effects — a correctness check for the interleaving.
    """
    if meta.get("order") != "interleaved":
        return

    _require_matplotlib()
    import matplotlib.pyplot as plt
    import numpy as np

    # Select up to 6 representative scenarios to keep the chart readable:
    # bs=1 decode, bs=32 decode, a prefill curve.
    picks = []
    for c in test_cases:
        if c["label"] in {
            "bs=1,  seq=1",
            "bs=32, seq=1",
            "bs=1,  seq=128",
            "bs=1,  seq=1024",
            "bs=4,  seq=256",
        }:
            picks.append(c)
    if not picks:
        return

    fig, ax = plt.subplots(figsize=(10, 5))

    for c in picks:
        label = c["label"]
        nf = all_samples[label]["nf"]
        f = all_samples[label]["f"]
        n = min(len(nf), len(f))
        if n == 0:
            continue
        delta = np.asarray(f[:n]) - np.asarray(nf[:n])
        ax.plot(range(n), delta, "o-", ms=3, lw=1.2, label=label)

    ax.axhline(0, color="black", lw=0.7)
    ax.set_xlabel("Round index")
    ax.set_ylabel("Fused − Non-Fused latency (ms)")
    ax.set_title(f"Per-Round Paired Delta\n{_title_suffix(meta)}", fontsize=11)
    ax.grid(linestyle="--", alpha=0.4)
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    _savefig(fig, out / "06_paired_delta", fmt)


# ── Chart 7: correctness info card ─────────────────────────────────


def _plot_correctness(verify_data, out, meta, fmt):
    _require_matplotlib()
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    passed = bool(verify_data.get("passed"))
    color = "#2ecc71" if passed else "#e74c3c"
    status = "PASS" if passed else "FAIL"

    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    ax.axis("off")

    ax.text(0.03, 0.86, "Correctness Verification", fontsize=14,
            fontweight="bold", transform=ax.transAxes)
    ax.text(0.03, 0.74, _title_suffix(meta), fontsize=9,
            color="#555", transform=ax.transAxes)

    rows = [
        ("Max |diff|",        f"{verify_data['max_diff']:.3e}"),
        ("Mean |diff|",       f"{verify_data['mean_diff']:.3e}"),
        ("Cosine similarity", f"{verify_data['cos_sim']:.8f}"),
    ]
    for i, (k, v) in enumerate(rows):
        y = 0.56 - i * 0.12
        ax.text(0.05, y, k, fontsize=10, transform=ax.transAxes)
        ax.text(0.55, y, v, fontsize=10, family="monospace",
                transform=ax.transAxes)

    ax.text(0.05, 0.12, "Status:", fontsize=11, fontweight="bold",
            transform=ax.transAxes)
    ax.text(0.22, 0.12, status, fontsize=12, fontweight="bold",
            color=color, transform=ax.transAxes)
    ax.text(0.40, 0.12, "(threshold  cos_sim > 0.999)", fontsize=9,
            color="#555", transform=ax.transAxes)

    ax.add_patch(Rectangle(
        (0.01, 0.02), 0.98, 0.96,
        linewidth=1.8, edgecolor=color, facecolor="none",
        transform=ax.transAxes,
    ))

    _savefig(fig, out / "07_correctness", fmt)


# ── Public entry point ─────────────────────────────────────────────


def save_e2e_plots(args, verify_data, all_samples, test_cases,
                   out_dir, fmt="png"):
    """
    Write the full chart suite into ``out_dir``.

    Parameters
    ----------
    args        : argparse namespace from bench_fused_ffn.py
    verify_data : dict or None — correctness verification result
    all_samples : ``{label: {"nf": [ms...], "f": [ms...]}}``
                  end-to-end latency samples per scenario
    test_cases  : the TEST_CASES list (for labels / shape metadata)
    out_dir     : target directory (created if missing)
    fmt         : one of "png" | "svg" | "pdf"

    Returns
    -------
    Path to the output directory.
    """
    _require_matplotlib()  # fail fast if matplotlib is missing

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    meta = _build_meta(args)

    _plot_speedup_overview(all_samples, test_cases, out, meta, fmt)
    _plot_nf_vs_f(all_samples, test_cases, out, meta, fmt)
    _plot_speedup_heatmap(all_samples, test_cases, out, meta, fmt)
    _plot_latency_distribution(all_samples, test_cases, out, meta, fmt)
    _plot_throughput_scaling(all_samples, test_cases, out, meta, fmt)
    _plot_paired_delta(all_samples, test_cases, out, meta, fmt)
    if verify_data:
        _plot_correctness(verify_data, out, meta, fmt)

    return out
