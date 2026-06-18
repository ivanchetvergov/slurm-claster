#!/usr/bin/env python3
import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.colors import ListedColormap
from scipy import stats
from stats_calculator import compute_stats
from plots import draw_elapsed, draw_log_error, draw_timeout_model, _set_time_ticks, _timeout_log_errors, _ORANGE_C

# ── style ─────────────────────────────────────────────────────────────────────

plt.rcParams.update({
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.alpha":        0.35,
    "grid.linestyle":    "--",
    "font.size":         10,
    "axes.titlesize":    11,
    "axes.labelsize":    10,
    "legend.fontsize":   9,
    "figure.facecolor":  "white",
    "axes.facecolor":    "#f8f8f8",
})

_BLUE  = "#4878CF"
_ORANGE = "#e87722"

# ── I/O ───────────────────────────────────────────────────────────────────────

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
    for col in ("ElapsedRaw", "TimelimitRaw", "ReqNodes"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def load_subsample(path: Path) -> pd.DataFrame:
    df = _load_csv(path)
    df["ElapsedRaw"]   = pd.to_numeric(df["ElapsedRaw"],   errors="coerce")
    df["TimelimitRaw"] = pd.to_numeric(df["TimelimitRaw"], errors="coerce")
    return df


def _save(fig, path: Path):
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Сохранено → {path}")


def _logloss(data: np.ndarray, pdf_fn) -> float:
    return float(-np.mean(np.log(np.maximum(pdf_fn(data), 1e-300))))


# ── nodelist parser ────────────────────────────────────────────────────────────

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


def _all_nodes(sim: pd.DataFrame) -> list[str]:
    valid = sim.dropna(subset=["NodeList"])
    return sorted({n for nl in valid["NodeList"] for n in _expand_nodelist(nl)})


# ── original_data.png ─────────────────────────────────────────────────────────

def plot_original_data(orig: pd.DataFrame, out: Path):
    from matplotlib.gridspec import GridSpec

    fig = plt.figure(figsize=(13, 12))
    fig.suptitle("Оригинальные данные  (UID=50109, sphere.slrm)", fontsize=13, y=0.98)
    gs = GridSpec(2, 2, figure=fig, height_ratios=[1.1, 1], hspace=0.45, wspace=0.35)

    ax_elapsed = fig.add_subplot(gs[0, :])
    ax_log_err = fig.add_subplot(gs[1, 0])
    ax_ratio   = fig.add_subplot(gs[1, 1])

    draw_elapsed(ax_elapsed, orig)
    draw_log_error(ax_log_err, orig)
    draw_timeout_model(ax_ratio, orig)

    _save(fig, out)


# ── sampled_data.png ──────────────────────────────────────────────────────────

def plot_sampled_data(orig: pd.DataFrame, sim: pd.DataFrame,
                      mu: float, sigma: float, scale: int, out: Path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Синтетические vs оригинальные данные", fontsize=13)

    orig_e = orig["ElapsedRaw"].dropna()
    orig_e = orig_e[orig_e > 0].to_numpy() / scale
    sim_e  = sim["ElapsedRaw"].dropna()
    sim_e  = sim_e[sim_e > 0].to_numpy()

    if len(sim_e) > 0:
        log_orig = np.log(orig_e)
        log_sim  = np.log(sim_e)
        lo = min(log_sim.min(), log_orig.min())
        hi = max(log_orig.max(), log_sim.max())
        x  = np.linspace(lo, hi, 400)

        kde_orig = stats.gaussian_kde(log_orig)
        kde_sim  = stats.gaussian_kde(log_sim)
        ll = _logloss(log_sim, kde_orig)

        ax1.hist(log_orig, bins=np.linspace(lo, hi, 70), density=True,
                 alpha=0.3, color=_BLUE)
        ax1.hist(log_sim,  bins=np.linspace(lo, hi, 40), density=True,
                 alpha=0.3, color=_ORANGE)
        ax1.plot(x, kde_orig(x), color=_BLUE,   lw=2, label=f"Оригинал / {scale}")
        ax1.plot(x, kde_sim(x),  color=_ORANGE, lw=2, label=f"Синтетика (n={len(sim_e)})")
        ax1.set_title(f"Elapsed  LL={ll:.3f}")
        _set_time_ticks(ax1, lo, hi)
    else:
        ax1.set_title("Elapsed (нет данных из sacct)")
    ax1.set_ylabel("Плотность")
    ax1.legend()

    # log_error: берём сэмплированное значение если есть (dry-run), иначе вычисляем
    if "LogError" in sim.columns:
        sim_err = pd.to_numeric(sim["LogError"], errors="coerce").dropna().to_numpy()
    else:
        sim_valid = sim[(sim["ElapsedRaw"] > 0) & (sim["TimelimitRaw"] > 0)]
        sim_err   = np.log(sim_valid["TimelimitRaw"] * 60 / sim_valid["ElapsedRaw"])
        sim_err   = sim_err[np.isfinite(sim_err)].to_numpy()

    completed  = orig[(orig["State"] == "COMPLETED") & (orig["ElapsedRaw"] > 0) & (orig["TimelimitRaw"] > 0)]
    orig_err_c = np.log(completed["TimelimitRaw"] * 60 / completed["ElapsedRaw"])
    orig_err_c = orig_err_c[np.isfinite(orig_err_c)].to_numpy()
    n_timeout  = int((orig["State"] == "TIMEOUT").sum())
    orig_err_t = _timeout_log_errors(n_timeout)
    orig_err   = np.concatenate([orig_err_c, orig_err_t])
    lo2 = -2.0
    hi2 = np.quantile(orig_err_c, 0.99)

    bins2 = np.linspace(lo2, hi2, 60)
    x2    = np.linspace(lo2, hi2, 300)
    ax2.hist([orig_err_c.clip(lo2, hi2), orig_err_t.clip(lo2, hi2)], bins=bins2,
             density=True, stacked=True, alpha=0.5,
             color=[_BLUE, _ORANGE_C],
             label=[f"Оригинал COMPLETED  n={len(orig_err_c)}",
                    f"Оригинал TIMEOUT  n={n_timeout} (модель)"])
    if len(sim_err) > 0:
        sim_clipped = sim_err.clip(lo2, hi2)
        kde_sim2 = stats.gaussian_kde(sim_clipped)
        ax2.plot(x2, kde_sim2(x2), color="green", lw=2,
                 label=f"Синтетика KDE  n={len(sim_err)}")
        ax2.fill_between(x2, kde_sim2(x2), alpha=0.12, color="green")
        ll2 = _logloss(sim_clipped, lambda x: stats.norm.pdf(x, mu, sigma))
        ax2.set_title(f"Log-error  LL={ll2:.3f}")
    else:
        ax2.set_title("Log-error")

    from stats_calculator import fit_gmm2
    from plots import gmm_pdf
    gw, gmu1, gs1, gmu2, gs2 = fit_gmm2(orig_err)
    x2_full = np.linspace(lo2, orig_err.max(), 400)
    ax2.plot(x2_full, gmm_pdf(x2_full, gw, gmu1, gs1, gmu2, gs2), "crimson", lw=2,
             label=f"GMM-2: {gw:.2f}·N({gmu1:.1f},{gs1:.1f}) + {1-gw:.2f}·N({gmu2:.1f},{gs2:.1f})")
    ax2.set_xticks(range(int(lo2), int(np.ceil(orig_err.max())) + 1, 2))
    ax2.set_xlabel("ln(Timelimit / Elapsed)")
    ax2.set_ylabel("Плотность")
    ax2.legend()

    plt.tight_layout()
    _save(fig, out)


# ── gantt_analysis.png ────────────────────────────────────────────────────────

def _compress_timeline(df: pd.DataFrame, gap_threshold_s: float = 300.0) -> pd.DataFrame:
    """Вырезает простои > gap_threshold_s и сдвигает последующие задачи влево."""
    df = df.copy().sort_values("Start")
    # строим покрытие: когда хоть одна задача работала
    events = sorted(
        [(row["Start"], +1, i) for i, row in df.iterrows()] +
        [(row["End"],   -1, i) for i, row in df.iterrows()]
    )
    shift = pd.Timedelta(0)
    active = 0
    idle_start = None
    shifts: list[tuple] = []  # (cutoff_time, shift_delta)

    for ts, delta, _ in events:
        if active == 0 and delta == +1 and idle_start is not None:
            gap = (ts - idle_start).total_seconds()
            if gap > gap_threshold_s:
                shift += pd.Timedelta(seconds=gap)
                shifts.append((ts, shift))
        active += delta
        if active == 0:
            idle_start = ts

    def apply_shift(t):
        s = pd.Timedelta(0)
        for cutoff, total in shifts:
            if t >= cutoff:
                s = total
        return t - s

    df["Start"] = df["Start"].apply(apply_shift)
    df["End"]   = df["End"].apply(apply_shift)
    return df


def _draw_gantt(ax, sim: pd.DataFrame, all_nodes: list[str], job_color: dict):
    valid = sim.dropna(subset=["Start", "End", "NodeList"]).copy()
    valid = valid[valid["Start"] < valid["End"]]
    valid = _compress_timeline(valid)
    t_min    = valid["Start"].min()
    node_idx = {n: i for i, n in enumerate(all_nodes)}

    for _, row in valid.iterrows():
        color = job_color[row["JobID"]]
        s = (row["Start"] - t_min).total_seconds() / 60
        e = (row["End"]   - t_min).total_seconds() / 60
        for node in _expand_nodelist(row["NodeList"]):
            ax.barh(node_idx[node], e - s, left=s, height=0.5,
                    color=color, alpha=0.88)

    ax.set_yticks(range(len(all_nodes)))
    ax.set_yticklabels(all_nodes)
    ax.set_xlabel("Время с начала, мин (простои исключены)")
    ax.set_title("Gantt: загрузка нод")
    ax.grid(axis="x", alpha=0.3, linestyle="--")


def _draw_heatmap(ax, sim: pd.DataFrame, all_nodes: list[str], job_ids: list, job_color: dict):
    valid = sim.dropna(subset=["Start", "End", "NodeList"]).copy()
    valid = valid[valid["Start"] < valid["End"]]

    t_min = valid["Start"].min()
    t_max = valid["End"].max()
    total_secs = (t_max - t_min).total_seconds()
    slot_secs  = max(20, int(total_secs / 200))
    n_slots    = int(total_secs / slot_secs) + 2

    job_idx  = {jid: i + 1 for i, jid in enumerate(job_ids)}
    node_idx = {n: i for i, n in enumerate(all_nodes)}
    grid     = np.zeros((len(all_nodes), n_slots), dtype=float)

    for _, row in valid.iterrows():
        t_s = max(0, int((row["Start"] - t_min).total_seconds() / slot_secs))
        t_e = min(n_slots, int((row["End"]   - t_min).total_seconds() / slot_secs) + 1)
        cidx = job_idx[row["JobID"]]
        for node in _expand_nodelist(row["NodeList"]):
            grid[node_idx[node], t_s:t_e] = cidx

    colors = ["#ececec"] + [job_color[jid] for jid in job_ids]
    cmap   = ListedColormap(colors)
    dur_min = total_secs / 60

    ax.imshow(grid, aspect="auto", cmap=cmap, vmin=0, vmax=len(job_ids),
              extent=[0, dur_min, len(all_nodes) - 0.5, -0.5],
              interpolation="nearest")
    ax.set_yticks(range(len(all_nodes)))
    ax.set_yticklabels(all_nodes)
    ax.set_xlabel("Время с начала, мин")
    ax.set_title("Heatmap: занятость нод (серый = простой)")
    ax.grid(False)


def plot_gantt_analysis(sim: pd.DataFrame, out: Path):
    valid = sim.dropna(subset=["Start", "End", "NodeList"]).copy()
    valid = valid[valid["Start"] < valid["End"]]
    if valid.empty:
        return

    all_nodes = _all_nodes(sim)
    job_ids   = sorted(valid["JobID"].unique())
    cmap_src  = plt.cm.tab20 if len(job_ids) > 10 else plt.cm.tab10
    job_color = {jid: cmap_src(i / max(len(job_ids), 10)) for i, jid in enumerate(job_ids)}

    fig, ax = plt.subplots(figsize=(14, max(4, len(all_nodes) * 0.9)))
    fig.suptitle("Gantt: загрузка кластера", fontsize=13)
    _draw_gantt(ax, sim, all_nodes, job_color)
    plt.tight_layout()
    _save(fig, out)


# ── slurm_analysis.png ────────────────────────────────────────────────────────

def _node_utilization(ax, sim: pd.DataFrame, all_nodes: list[str]):
    valid = sim.dropna(subset=["Start", "End", "NodeList"]).copy()
    valid = valid[valid["Start"] < valid["End"]]
    total = len(all_nodes)
    times = sorted(set(valid["Start"].tolist() + valid["End"].tolist()))
    ts, util = [], []
    for t in times:
        busy = sum(
            1 for nl in valid[(valid["Start"] <= t) & (valid["End"] > t)]["NodeList"]
            for _ in _expand_nodelist(nl)
        )
        ts.append(t); util.append(busy / total * 100)

    ax.step(ts, util, where="post", lw=1.5, color=_BLUE)
    ax.fill_between(ts, util, step="post", alpha=0.2, color=_BLUE)
    ax.set_ylim(0, 115)
    ax.axhline(100, color="crimson", lw=0.8, ls="--", alpha=0.6)
    ax.set_ylabel("Занято нод, %")
    ax.set_title(f"Утилизация нод (всего: {total})")
    ax.xaxis_date()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")


def _queue_depth(ax, sim: pd.DataFrame):
    valid = sim.dropna(subset=["Submit", "Start", "End"]).copy()
    t_min = valid["Submit"].min()
    times = sorted(set(valid["Submit"].tolist() + valid["Start"].tolist() + valid["End"].tolist()))
    rel   = [(t - t_min).total_seconds() / 60 for t in times]
    depths = [int(((valid["Submit"] <= t) & (valid["Start"] > t)).sum()) for t in times]

    ax.step(rel, depths, where="post", lw=1.5, color=_ORANGE)
    ax.fill_between(rel, depths, step="post", alpha=0.2, color=_ORANGE)
    ax.set_xlabel("Время с начала, мин")
    ax.set_ylabel("Задач в очереди")
    ax.set_title("Глубина очереди")
    ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))


def _wait_time(ax, sim: pd.DataFrame):
    valid = sim.dropna(subset=["Submit", "Start"]).copy()
    valid = valid.sort_values("Submit").reset_index(drop=True)
    wait  = (valid["Start"] - valid["Submit"]).dt.total_seconds() / 60
    x     = range(len(wait))

    ax.bar(x, wait, color=_BLUE, alpha=0.75, width=0.6)
    ax.set_xticks(list(x))
    ax.set_xticklabels(valid["JobID"].astype(str), rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("Ожидание, мин")
    ax.set_title("Время ожидания (по порядку отправки)")


def _scatter_nodes_wait(ax, sim: pd.DataFrame):
    valid = sim.dropna(subset=["Submit", "Start", "ReqNodes"]).copy()
    valid = valid[valid["ReqNodes"] > 0]
    wait  = (valid["Start"] - valid["Submit"]).dt.total_seconds() / 60
    nodes = valid["ReqNodes"].astype(int)
    rng   = np.random.default_rng(42)
    jitter = rng.uniform(-0.15, 0.15, len(nodes))

    ax.scatter(nodes + jitter, wait, alpha=0.7, s=50, color=_BLUE, edgecolors="white", lw=0.5)

    for n in sorted(nodes.unique()):
        m = wait[nodes == n].mean()
        ax.hlines(m, n - 0.35, n + 0.35, colors=_ORANGE, lw=2)

    ax.set_xticks(sorted(nodes.unique()))
    ax.set_xlabel("Запрошено нод")
    ax.set_ylabel("Ожидание, мин")
    ax.set_title("Ожидание vs число нод  (— среднее)")


def _timelimit_efficiency(ax, sim: pd.DataFrame):
    valid = sim[(sim["ElapsedRaw"] > 0) & (sim["TimelimitRaw"] > 0)].copy()
    eff   = valid["ElapsedRaw"] / (valid["TimelimitRaw"] * 60) * 100
    valid = valid.copy(); valid["eff"] = eff
    valid = valid.sort_values("eff", ascending=False)

    ax.bar(range(len(valid)), valid["eff"], color=_BLUE, alpha=0.75, width=0.6)
    ax.axhline(100, color="crimson", lw=1, ls="--", alpha=0.7)
    ax.set_ylabel("Использовано, %")
    ax.set_title("Сколько задача использовала от выделенного времени, %")
    ax.set_xticks(range(len(valid)))
    ax.set_xticklabels(valid["JobID"].astype(str), rotation=45, ha="right", fontsize=7)
    ax.set_xlabel("Job ID")


def plot_slurm_analysis(sim: pd.DataFrame, out: Path):
    from matplotlib.gridspec import GridSpec

    all_nodes = _all_nodes(sim)
    valid     = sim.dropna(subset=["Start", "End", "NodeList"]).copy()
    valid     = valid[valid["Start"] < valid["End"]]
    job_ids   = sorted(valid["JobID"].unique())
    cmap_src  = plt.cm.tab20 if len(job_ids) > 10 else plt.cm.tab10
    job_color = {jid: cmap_src(i / max(len(job_ids), 10)) for i, jid in enumerate(job_ids)}

    fig = plt.figure(figsize=(16, 12))
    fig.suptitle("Анализ работы SLURM", fontsize=14, y=0.98)
    gs = GridSpec(3, 2, figure=fig, height_ratios=[0.8, 1.3, 1.3],
                  hspace=0.55, wspace=0.35)

    ax_gantt   = fig.add_subplot(gs[0, :])   # Gantt — весь верхний ряд
    ax_queue   = fig.add_subplot(gs[1, 0])
    ax_scatter = fig.add_subplot(gs[1, 1])
    ax_slow    = fig.add_subplot(gs[2, 0])
    ax_eff     = fig.add_subplot(gs[2, 1])

    _draw_gantt(ax_gantt, sim, all_nodes, job_color)
    _queue_depth(ax_queue, sim)
    _scatter_nodes_wait(ax_scatter, sim)
    _wait_time(ax_slow, sim)
    _timelimit_efficiency(ax_eff, sim)

    _save(fig, out)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    args    = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(exist_ok=True)

    sim  = load_sacct(Path(args.sacct))
    orig = load_subsample(Path(args.subsample))
    ref  = compute_stats(orig)
    mu, sigma = ref.log_error_mu, ref.log_error_sigma

    print(f"Загружено из sacct: {len(sim)} записей")

    plot_original_data(orig, out_dir / "original_data.png")
    plot_sampled_data(orig, sim, mu, sigma, args.scale, out_dir / "sampled_data.png")
    plot_gantt_analysis(sim, out_dir / "gantt_analysis.png")
    plot_slurm_analysis(sim, out_dir / "slurm_analysis.png")


if __name__ == "__main__":
    main()
