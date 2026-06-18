#!/usr/bin/env python3
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats as sp
from scipy.optimize import minimize

from stats_calculator import compute_stats, fit_gmm2
from data_loader import SlurmDataLoader


_ORANGE_C = "#e87722"

TIMEOUT_FACTOR_MU    = 2.0   # центр модели недооценки TIMEOUT (в разах)
TIMEOUT_FACTOR_SIGMA = 1.0   # разброс
TIMEOUT_FACTOR_LO    = 1.2   # нижняя граница: хотя бы в 1.2 раз недооценили
TIMEOUT_FACTOR_HI    = 5.0   # верхняя граница

DATA_DIR  = Path(__file__).parent.parent / "data"
PLOTS_DIR = Path(__file__).parent.parent / "plots"


def _parse_args():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--scale", type=int, default=1000)
    p.add_argument("--n",     type=int, default=50)
    p.add_argument("--seed",  type=int, default=42)
    p.add_argument("--uid",   type=int, default=50109)
    p.add_argument("--job",   default="sphere.slrm")
    return p.parse_args()


def load_orig(uid: int, job: str) -> pd.DataFrame:
    df = SlurmDataLoader(DATA_DIR / f"cleared_{uid}_{job}.csv").load().df
    df["ElapsedRaw"]   = pd.to_numeric(df["ElapsedRaw"],   errors="coerce")
    df["TimelimitRaw"] = pd.to_numeric(df["TimelimitRaw"], errors="coerce")
    return df


def logloss(data: np.ndarray, pdf_fn) -> float:
    return float(-np.mean(np.log(np.maximum(pdf_fn(data), 1e-300))))



def gmm_pdf(x, w, mu1, s1, mu2, s2) -> np.ndarray:
    return w * sp.norm.pdf(x, mu1, s1) + (1 - w) * sp.norm.pdf(x, mu2, s2)


def gmm_cdf(x, w, mu1, s1, mu2, s2) -> np.ndarray:
    return w * sp.norm.cdf(x, mu1, s1) + (1 - w) * sp.norm.cdf(x, mu2, s2)


_TIME_TICKS = [
    (0.01,      "0.01с"),
    (0.1,       "0.1с"),
    (0.5,       "0.5с"),
    (1,         "1с"),
    (5,         "5с"),
    (30,        "30с"),
    (60,        "1м"),
    (300,       "5м"),
    (600,       "10м"),
    (3_600,     "1ч"),
    (21_600,    "6ч"),
    (86_400,    "1д"),
    (7*86_400,  "1нед"),
    (30*86_400, "1мес"),
]

def _set_time_ticks(ax, lo: float, hi: float):
    ticks = [(lv, lbl) for v, lbl in _TIME_TICKS if lo <= (lv := np.log(v)) <= hi]
    if ticks:
        ax.set_xticks([t for t, _ in ticks])
        ax.set_xticklabels([lbl for _, lbl in ticks])
    ax.set_xlim(lo - 0.3, hi + 0.3)
    ax.set_xlabel("Время выполнения")


# ── ax-level drawing functions (reusable) ─────────────────────────────────────

def draw_elapsed(ax, orig: pd.DataFrame):
    mask_c = orig["State"].isin(["COMPLETED"]) & (orig["ElapsedRaw"] > 0)
    mask_t = orig["State"].isin(["TIMEOUT"])   & (orig["ElapsedRaw"] > 0)
    e_c = orig.loc[mask_c, "ElapsedRaw"].dropna().to_numpy()
    e_t = orig.loc[mask_t, "ElapsedRaw"].dropna().to_numpy()
    e   = np.concatenate([e_c, e_t])
    log_orig = np.log(e)
    log_c    = np.log(e_c)
    log_t    = np.log(e_t) if len(e_t) else np.array([])
    lo_e, hi_e = log_orig.min(), log_orig.max()
    bins = np.linspace(lo_e, hi_e, 120)
    ctrs = (bins[:-1] + bins[1:]) / 2
    median_h = np.exp(float(log_orig.mean())) / 3600

    mu_tn  = (lo_e + hi_e) / 2
    sig_tn = (hi_e - lo_e) / 6
    a_tn   = (lo_e - mu_tn) / sig_tn
    b_tn   = (hi_e - mu_tn) / sig_tn
    pdf_tn   = sp.truncnorm.pdf(ctrs, a_tn, b_tn, loc=mu_tn, scale=sig_tn)
    ks_tn, _ = sp.kstest(log_orig, sp.truncnorm.cdf, args=(a_tn, b_tn, mu_tn, sig_tn))
    ll_tn    = logloss(log_orig, lambda x: sp.truncnorm.pdf(x, a_tn, b_tn, loc=mu_tn, scale=sig_tn))

    gw, gmu1, gs1, gmu2, gs2 = fit_gmm2(log_orig)
    pdf_gmm   = gmm_pdf(ctrs, gw, gmu1, gs1, gmu2, gs2)
    ks_gmm, _ = sp.kstest(log_orig, lambda x: gmm_cdf(x, gw, gmu1, gs1, gmu2, gs2))
    ll_gmm    = logloss(log_orig, lambda x: gmm_pdf(x, gw, gmu1, gs1, gmu2, gs2))

    if len(log_t):
        ax.hist([log_c, log_t], bins=bins, density=True, stacked=True, alpha=0.6,
                color=["steelblue", _ORANGE_C],
                label=[f"COMPLETED  n={len(e_c)}", f"TIMEOUT  n={len(e_t)}"])
    else:
        ax.hist(log_c, bins=bins, density=True, alpha=0.6, color="steelblue",
                label=f"данные  n={len(e_c)}")
    ax.plot(ctrs, pdf_tn, "orange", lw=2,
            label=f"усечённое норм. ★  KS={ks_tn:.3f}  LL={ll_tn:.3f}")
    ax.plot(ctrs, pdf_gmm, "g--", lw=1.5,
            label=f"GMM-2  KS={ks_gmm:.3f}  LL={ll_gmm:.3f}")
    _set_time_ticks(ax, lo_e, hi_e)
    ax.set_ylabel("Плотность")
    ax.set_title(f"Фактическое время выполнения  (лог. шкала, медиана ≈ {median_h:.1f} ч)")
    ax.legend(fontsize=9, framealpha=0.8)


def _timeout_log_errors(n: int) -> np.ndarray:
    """Импутация log_error для TIMEOUT-задач через модель недооценки.

    factor ~ TruncNorm(mu, sigma) ∈ [LO, HI]  →  log_error = -ln(factor) ∈ [-ln(HI), 0]
    """
    a = (TIMEOUT_FACTOR_LO - TIMEOUT_FACTOR_MU) / TIMEOUT_FACTOR_SIGMA
    b = (TIMEOUT_FACTOR_HI - TIMEOUT_FACTOR_MU) / TIMEOUT_FACTOR_SIGMA
    factors = sp.truncnorm.rvs(a, b, loc=TIMEOUT_FACTOR_MU, scale=TIMEOUT_FACTOR_SIGMA,
                               size=n, random_state=42)
    return -np.log(factors)


def draw_log_error(ax, orig: pd.DataFrame):
    valid_mask = (orig["ElapsedRaw"] > 0) & (orig["TimelimitRaw"] > 0)
    completed  = orig[valid_mask & (orig["State"] == "COMPLETED")]
    log_err    = np.log(completed["TimelimitRaw"] * 60 / completed["ElapsedRaw"])
    log_err    = log_err[np.isfinite(log_err)].to_numpy()

    n_completed = len(log_err)
    n_timeout   = int((orig["State"] == "TIMEOUT").sum())
    log_err     = np.concatenate([log_err, _timeout_log_errors(n_timeout)])

    lo, hi = log_err.min(), log_err.max()
    data   = log_err
    bins   = np.linspace(lo, hi, 50)

    norm_params = sp.norm.fit(data)
    ks_n, _     = sp.kstest(data, sp.norm.cdf, args=norm_params)
    ll_n        = logloss(data, lambda x: sp.norm.pdf(x, *norm_params))

    gw, gmu1, gs1, gmu2, gs2 = fit_gmm2(data)
    ks_g, _ = sp.kstest(data, lambda x: gmm_cdf(x, gw, gmu1, gs1, gmu2, gs2))
    ll_g    = logloss(data, lambda x: gmm_pdf(x, gw, gmu1, gs1, gmu2, gs2))

    log_err_timeout = _timeout_log_errors(n_timeout)
    c_clipped = log_err[:n_completed]
    t_clipped = log_err_timeout

    x = np.linspace(lo, hi, 400)
    ax.hist([c_clipped, t_clipped], bins=bins, density=True, stacked=True, alpha=0.6,
            color=["steelblue", _ORANGE_C],
            label=[f"COMPLETED  n={n_completed}", f"TIMEOUT  n={n_timeout} (модель)"])
    ax.plot(x, sp.norm.pdf(x, *norm_params), "orange", lw=2,
            label=f"N(μ={norm_params[0]:.2f}, σ={norm_params[1]:.2f}) ★  KS={ks_n:.3f}  LL={ll_n:.3f}")
    ax.plot(x, gmm_pdf(x, gw, gmu1, gs1, gmu2, gs2), "g--", lw=1.5, alpha=0.85,
            label=f"GMM-2: {gw:.2f}·N({gmu1:.1f},{gs1:.1f}) + {1-gw:.2f}·N({gmu2:.1f},{gs2:.1f})  KS={ks_g:.3f}  LL={ll_g:.3f}")
    ax.set_ylim(0, 0.40)
    ax.yaxis.set_major_locator(plt.MultipleLocator(0.10))
    ax.set_xticks(range(int(np.floor(lo)), int(np.ceil(hi)) + 1))
    ax.set_xlabel("ln(timelimit / elapsed)")
    ax.set_ylabel("Плотность")
    ax.set_title("Логарифм ошибки оценки времени  ln(timelimit / elapsed)")
    ax.legend(fontsize=8, framealpha=0.8)


def draw_timeout_model(ax, orig: pd.DataFrame):
    """Модель недооценки для TIMEOUT: factor = true_elapsed / timelimit ~ TruncNorm ∈ [1, 5]."""
    n_timeout = int((orig["State"] == "TIMEOUT").sum())

    a = (TIMEOUT_FACTOR_LO - TIMEOUT_FACTOR_MU) / TIMEOUT_FACTOR_SIGMA
    b = (TIMEOUT_FACTOR_HI - TIMEOUT_FACTOR_MU) / TIMEOUT_FACTOR_SIGMA
    x = np.linspace(TIMEOUT_FACTOR_LO, TIMEOUT_FACTOR_HI, 300)
    pdf = sp.truncnorm.pdf(x, a, b, loc=TIMEOUT_FACTOR_MU, scale=TIMEOUT_FACTOR_SIGMA)

    samples = sp.truncnorm.rvs(a, b, loc=TIMEOUT_FACTOR_MU, scale=TIMEOUT_FACTOR_SIGMA,
                               size=max(n_timeout * 50, 500), random_state=42)

    ax.hist(samples, bins=20, density=True, alpha=0.45, color=_ORANGE_C,
            label=f"сэмплы  (реальных TIMEOUT: {n_timeout})")
    ax.plot(x, pdf, color="crimson", lw=2,
            label=f"TruncNorm(μ={TIMEOUT_FACTOR_MU}, σ={TIMEOUT_FACTOR_SIGMA})  ∈ [1.2, 5]")
    ax.set_xticks([1.2, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0])
    ax.set_xlabel("true_elapsed / timelimit")
    ax.set_ylabel("Плотность")
    ax.set_title("Модель недооценки TIMEOUT-задач")
    ax.legend(fontsize=9, framealpha=0.8)


def plot_original_data(orig: pd.DataFrame):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 10))
    fig.suptitle("Оригинальные данные  (UID=50109, sphere.slrm)", fontsize=13)
    draw_elapsed(ax1, orig)
    draw_log_error(ax2, orig)
    plt.tight_layout()
    fig.subplots_adjust(top=0.97, hspace=0.18)
    out = PLOTS_DIR / "original_data.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Сохранено → {out}")


def main():
    args = _parse_args()
    orig = load_orig(args.uid, args.job)
    ref = compute_stats(orig)
    print(ref)
    PLOTS_DIR.mkdir(exist_ok=True)
    plot_original_data(orig)


if __name__ == "__main__":
    main()
