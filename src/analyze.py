#!/usr/bin/env python3
import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy import stats
from stats_calculator import compute_stats
from plots import draw_elapsed, draw_log_error


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--sacct",      required=True)
    p.add_argument("--subsample",  required=True)
    p.add_argument("--scale",      type=int, default=1)
    p.add_argument("--output-dir", default="plots")
    return p.parse_args()


def _load_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="|", low_memory=False)
    df.columns = df.columns.str.strip()
    return df


def load_sacct(path: Path) -> pd.DataFrame:
    df = _load_csv(path)
    for col in ("Submit", "Start", "End"):
        df[col] = pd.to_datetime(df[col], errors="coerce")
    for col in ("ElapsedRaw", "TimelimitRaw"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def load_subsample(path: Path) -> pd.DataFrame:
    df = _load_csv(path)
    df["ElapsedRaw"]   = pd.to_numeric(df["ElapsedRaw"],   errors="coerce")
    df["TimelimitRaw"] = pd.to_numeric(df["TimelimitRaw"], errors="coerce")
    return df


def _logloss(data: np.ndarray, pdf_fn) -> float:
    return float(-np.mean(np.log(np.maximum(pdf_fn(data), 1e-300))))


def _save(fig, path: Path):
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Сохранено → {path}")


# ── original_data.png ─────────────────────────────────────────────────────────

def plot_original_data(orig: pd.DataFrame, out: Path):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 10))
    fig.suptitle("Оригинальные данные  (UID=50109, sphere.slrm)", fontsize=13)
    draw_elapsed(ax1, orig)
    draw_log_error(ax2, orig)
    plt.tight_layout()
    _save(fig, out)


# ── sampled_data.png ──────────────────────────────────────────────────────────

def plot_sampled_data(orig: pd.DataFrame, sim: pd.DataFrame,
                      mu: float, sigma: float, scale: int, out: Path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Синтетические vs оригинальные данные", fontsize=13)

    # elapsed comparison
    orig_e = orig["ElapsedRaw"].dropna()
    orig_e = orig_e[orig_e > 0].to_numpy() / scale
    sim_e  = sim["ElapsedRaw"].dropna()
    sim_e  = sim_e[sim_e > 0].to_numpy()

    if len(sim_e) > 0:
        log_orig = np.log(orig_e)
        log_sim  = np.log(sim_e)
        lo = min(log_orig.min(), log_sim.min())
        hi = max(log_orig.max(), log_sim.max())
        bins = np.linspace(lo, hi, 35)
        kde_orig = stats.gaussian_kde(log_orig)
        ll = _logloss(log_sim, kde_orig)
        ax1.hist(log_orig, bins=bins, density=True, alpha=0.55, label=f"Оригинал / {scale}")
        ax1.hist(log_sim,  bins=bins, density=True, alpha=0.55, label=f"Синтетика (n={len(sim_e)})")
        ax1.set_title(f"Elapsed  LL={ll:.3f}")
    else:
        ax1.set_title("Elapsed (нет данных из sacct)")
    ax1.set_xlabel("ln(Elapsed)")
    ax1.set_ylabel("Плотность")
    ax1.legend(fontsize=9)

    # log_error comparison
    sim_valid = sim[(sim["ElapsedRaw"] > 0) & (sim["TimelimitRaw"] > 0)]
    sim_err = np.log(sim_valid["TimelimitRaw"] * 60 / sim_valid["ElapsedRaw"])
    sim_err = sim_err[np.isfinite(sim_err)].to_numpy()

    orig_valid = orig[(orig["ElapsedRaw"] > 0) & (orig["TimelimitRaw"] > 0)]
    orig_err = np.log(orig_valid["TimelimitRaw"] * 60 / orig_valid["ElapsedRaw"])
    orig_err = orig_err[np.isfinite(orig_err)].to_numpy()
    lo2, hi2 = np.quantile(orig_err, 0.01), np.quantile(orig_err, 0.99)

    bins2 = np.linspace(lo2, hi2, 35)
    x = np.linspace(lo2, hi2, 300)
    ax2.hist(orig_err.clip(lo2, hi2), bins=bins2, density=True, alpha=0.55,
             label="Оригинал")
    if len(sim_err) > 0:
        ax2.hist(sim_err.clip(lo2, hi2), bins=bins2, density=True, alpha=0.55,
                 label=f"Синтетика (n={len(sim_err)})")
        ll2 = _logloss(sim_err.clip(lo2, hi2), lambda x: stats.norm.pdf(x, mu, sigma))
        ax2.set_title(f"Log-error  LL={ll2:.3f}")
    else:
        ax2.set_title("Log-error")
    ax2.plot(x, stats.norm.pdf(x, mu, sigma), "r-", lw=1.5,
             label=f"N(μ={mu:.2f}, σ={sigma:.2f})")
    ax2.set_xlabel("ln(Timelimit / Elapsed)")
    ax2.set_ylabel("Плотность")
    ax2.legend(fontsize=9)

    _save(fig, out)


# ── slurm_analysis.png ────────────────────────────────────────────────────────

def _expand_nodelist(nodelist: str) -> list[str]:
    nodes = []
    for part in re.split(r",(?![^\[]*\])", nodelist):
        m = re.match(r"^(.*?)\[(.+)\]$", part)
        if m:
            prefix = m.group(1)
            for r in m.group(2).split(","):
                if "-" in r:
                    a, b = r.split("-")
                    nodes.extend(f"{prefix}{i}" for i in range(int(a), int(b) + 1))
                else:
                    nodes.append(f"{prefix}{r}")
        else:
            nodes.append(part)
    return nodes


def _gantt(ax, sim: pd.DataFrame):
    valid = sim.dropna(subset=["Start", "End", "NodeList"]).copy()
    valid = valid[valid["Start"] < valid["End"]]
    if valid.empty:
        ax.set_title("Gantt: нет данных")
        return

    cmap = plt.cm.tab10.colors
    job_color = {jid: cmap[i % len(cmap)] for i, jid in enumerate(sorted(valid["JobID"].unique()))}
    all_nodes = sorted({n for nl in valid["NodeList"] for n in _expand_nodelist(nl)})
    node_idx = {n: i for i, n in enumerate(all_nodes)}

    for _, row in valid.iterrows():
        color = job_color[row["JobID"]]
        s = mdates.date2num(row["Start"])
        e = mdates.date2num(row["End"])
        for node in _expand_nodelist(row["NodeList"]):
            ax.barh(node_idx[node], e - s, left=s, height=0.4, color=color, alpha=0.85)

    ax.set_yticks(range(len(all_nodes)))
    ax.set_yticklabels(all_nodes, fontsize=8)
    ax.xaxis_date()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")
    ax.set_title("Gantt: загрузка нод")
    ax.set_xlabel("Время")


def _node_utilization(ax, sim: pd.DataFrame):
    valid = sim.dropna(subset=["Start", "End", "NodeList"]).copy()
    valid = valid[valid["Start"] < valid["End"]]
    if valid.empty:
        ax.set_title("Утилизация: нет данных")
        return

    all_nodes = {n for nl in valid["NodeList"] for n in _expand_nodelist(nl)}
    total = len(all_nodes)
    times = sorted(set(valid["Start"].tolist() + valid["End"].tolist()))
    ts, util = [], []
    for t in times:
        busy = sum(
            1 for nl in valid[(valid["Start"] <= t) & (valid["End"] > t)]["NodeList"]
            for _ in _expand_nodelist(nl)
        )
        ts.append(t)
        util.append(busy / total * 100)

    ax.step(ts, util, where="post", lw=1.5, color="steelblue")
    ax.fill_between(ts, util, step="post", alpha=0.25, color="steelblue")
    ax.set_ylim(0, 110)
    ax.set_ylabel("Занято нод, %")
    ax.set_title(f"Утилизация нод (всего: {total})")
    ax.xaxis_date()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")
    ax.grid(True, alpha=0.3)


def _wait_time(ax, sim: pd.DataFrame):
    valid = sim.dropna(subset=["Submit", "Start"]).copy()
    valid = valid.sort_values("Submit").reset_index(drop=True)
    wait = (valid["Start"] - valid["Submit"]).dt.total_seconds()
    jobs = valid["JobID"].astype(str)

    ax.bar(range(len(wait)), wait, color="steelblue", alpha=0.75, width=0.5)
    ax.set_xticks(range(len(wait)))
    ax.set_xticklabels(jobs, fontsize=8)
    ax.set_xlabel("JobID")
    ax.set_ylabel("Ожидание, с")
    ax.set_title("Время ожидания в очереди")
    ax.grid(True, alpha=0.3, axis="y")


def _timelimit_efficiency(ax, sim: pd.DataFrame):
    valid = sim[(sim["ElapsedRaw"] > 0) & (sim["TimelimitRaw"] > 0)].copy()
    eff = valid["ElapsedRaw"] / (valid["TimelimitRaw"] * 60) * 100
    jobs = valid["JobID"].astype(str)

    bars = ax.bar(range(len(eff)), eff, color="steelblue", alpha=0.75, width=0.5)
    ax.axhline(100, color="r", lw=1, ls="--", alpha=0.6)
    ax.set_xticks(range(len(eff)))
    ax.set_xticklabels(jobs, fontsize=8)
    ax.set_ylim(0, max(eff.max() * 1.2, 110))
    ax.set_xlabel("JobID")
    ax.set_ylabel("Использовано, %")
    ax.set_title("Использование запрошенного времени")
    ax.grid(True, alpha=0.3, axis="y")


def plot_slurm_analysis(sim: pd.DataFrame, out: Path):
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle("Анализ работы SLURM", fontsize=13)

    _gantt(axes[0, 0], sim)
    _node_utilization(axes[0, 1], sim)
    _wait_time(axes[1, 0], sim)
    _timelimit_efficiency(axes[1, 1], sim)

    plt.tight_layout()
    _save(fig, out)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(exist_ok=True)

    sim  = load_sacct(Path(args.sacct))
    orig = load_subsample(Path(args.subsample))
    ref  = compute_stats(orig)
    mu, sigma = ref.log_error_mu, ref.log_error_sigma

    print(f"Загружено из sacct: {len(sim)} записей")

    plot_original_data(orig, out_dir / "original_data.png")
    plot_sampled_data(orig, sim, mu, sigma, args.scale, out_dir / "sampled_data.png")
    plot_slurm_analysis(sim, out_dir / "slurm_analysis.png")


if __name__ == "__main__":
    main()
