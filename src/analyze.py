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


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--sacct",     required=True, help="sacct_results.csv")
    p.add_argument("--subsample", required=True, help="cleared_<uid>_<job>.csv")
    p.add_argument("--scale",     type=int, default=1,
                   help="масштабный коэффициент, использованный при генерации")
    p.add_argument("--output",    default="analysis.png")
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


def plot_elapsed_hist(ax, orig: pd.DataFrame, sim: pd.DataFrame, scale: int):
    orig_e = orig["ElapsedRaw"].dropna()
    orig_e = orig_e[orig_e > 0].to_numpy() / scale
    sim_e  = sim["ElapsedRaw"].dropna()
    sim_e  = sim_e[sim_e > 0].to_numpy()

    if len(sim_e) == 0:
        ax.set_title("ElapsedRaw (нет данных из sacct)")
        return

    log_orig = np.log(orig_e)
    log_sim  = np.log(sim_e)
    lo = min(log_orig.min(), log_sim.min())
    hi = max(log_orig.max(), log_sim.max())
    bins = np.linspace(lo, hi, 35)

    kde_orig = stats.gaussian_kde(log_orig)
    ll = _logloss(log_sim, kde_orig)

    ax.hist(log_orig, bins=bins, density=True, alpha=0.55, label=f"Оригинал / {scale}")
    ax.hist(log_sim,  bins=bins, density=True, alpha=0.55, label="Синтетика (sacct)")
    ax.set_xlabel("ln(ElapsedRaw)")
    ax.set_ylabel("Плотность")
    ax.set_title(f"Распределение времени выполнения  LL={ll:.3f}")
    ax.legend()


def plot_log_error(ax, sim: pd.DataFrame, mu: float, sigma: float):
    valid = sim[(sim["ElapsedRaw"] > 0) & (sim["TimelimitRaw"] > 0)].copy()
    log_err = np.log(valid["TimelimitRaw"] * 60 / valid["ElapsedRaw"])
    log_err = log_err[np.isfinite(log_err)].to_numpy()

    if len(log_err) == 0:
        ax.set_title("Ошибка оценки времени (нет данных)")
        return

    lo, hi = np.quantile(log_err, 0.02), np.quantile(log_err, 0.98)
    data = log_err.clip(lo, hi)
    bins = np.linspace(lo, hi, 25)

    ll = _logloss(data, lambda x: stats.norm.pdf(x, mu, sigma))

    ax.hist(data, bins=bins, density=True, alpha=0.7, label="Данные (sacct)")
    x = np.linspace(lo, hi, 300)
    ax.plot(x, stats.norm.pdf(x, mu, sigma), "r-", lw=2,
            label=f"N(μ={mu:.2f}, σ={sigma:.2f})")
    ax.set_xlabel("ln(Timelimit / Elapsed)")
    ax.set_ylabel("Плотность")
    ax.set_title(f"Ошибка оценки времени  LL={ll:.3f}")
    ax.legend()


def plot_wait_time(ax, sim: pd.DataFrame):
    valid = sim.dropna(subset=["Submit", "Start"]).copy()
    valid = valid.sort_values("Submit").reset_index(drop=True)
    wait = (valid["Start"] - valid["Submit"]).dt.total_seconds()

    ax.plot(range(1, len(wait) + 1), wait, "o-", ms=4, lw=1)
    ax.set_xlabel("Порядковый номер заявки (по времени отправки)")
    ax.set_ylabel("Ожидание, с")
    ax.set_title("Время ожидания в очереди")
    ax.grid(True, alpha=0.3)


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


def plot_gantt(ax, sim: pd.DataFrame):
    valid = sim.dropna(subset=["Start", "End", "NodeList"]).copy()
    valid = valid[valid["Start"] < valid["End"]]

    if valid.empty:
        ax.set_title("Gantt: нет данных")
        return

    cmap = plt.cm.tab10.colors
    job_ids = sorted(valid["JobID"].unique())
    job_color = {jid: cmap[i % len(cmap)] for i, jid in enumerate(job_ids)}

    all_nodes = sorted({n for nl in valid["NodeList"] for n in _expand_nodelist(nl)})
    node_idx = {n: i for i, n in enumerate(all_nodes)}

    seen = set()
    for _, row in valid.iterrows():
        jid = row["JobID"]
        color = job_color[jid]
        start = mdates.date2num(row["Start"])
        end = mdates.date2num(row["End"])
        for node in _expand_nodelist(row["NodeList"]):
            label = jid if jid not in seen else None
            seen.add(jid)
            ax.barh(node_idx[node], end - start, left=start, height=0.6,
                    color=color, alpha=0.85, label=label)

    ax.set_yticks(range(len(all_nodes)))
    ax.set_yticklabels(all_nodes)
    ax.xaxis_date()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")
    ax.set_xlabel("Время")
    ax.set_title("Gantt: загрузка нод")
    ax.legend(title="JobID", fontsize=7, loc="upper left")


def main():
    args = parse_args()

    sim = load_sacct(Path(args.sacct))
    orig = load_subsample(Path(args.subsample))
    ref = compute_stats(orig)
    mu, sigma = ref.log_error_mu, ref.log_error_sigma

    print(f"Загружено из sacct: {len(sim)} записей")
    print(f"Параметры из оригинала: mu={mu:.4f}, sigma={sigma:.4f}")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Анализ результатов имитатора SLURM", fontsize=14)

    plot_elapsed_hist(axes[0, 0], orig, sim, args.scale)
    plot_log_error(axes[0, 1], sim, mu, sigma)
    plot_wait_time(axes[1, 0], sim)
    plot_gantt(axes[1, 1], sim)

    plt.tight_layout()
    out = Path(args.output)
    plt.savefig(out, dpi=150)
    print(f"Сохранено → {out}")


if __name__ == "__main__":
    main()
