#!/usr/bin/env python3
"""
Автономный анализ варианта UID=50109, sphere.slrm.
Выходные файлы: plots/elapsed.png, plots/log_error.png
"""
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

from stats_calculator import compute_stats
from data_loader import SlurmDataLoader


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
    """Отрицательная логправдоподобность на выборке: -1/n Σ log f(xᵢ)."""
    return float(-np.mean(np.log(np.maximum(pdf_fn(data), 1e-300))))


def fit_gmm2(data: np.ndarray):
    """MLE для смеси двух нормальных. Возвращает (w, mu1, s1, mu2, s2)."""
    def neg_ll(p):
        w, mu1, s1, mu2, s2 = p
        if not (0.05 < w < 0.95 and s1 > 0.05 and s2 > 0.05):
            return 1e10
        pdf = w * sp.norm.pdf(data, mu1, s1) + (1 - w) * sp.norm.pdf(data, mu2, s2)
        return -np.sum(np.log(np.maximum(pdf, 1e-300)))

    q    = np.percentile(data, [15, 85])
    span = float(data.std())
    inits = [
        [0.5, q[0], span / 3, q[1], span / 3],
        [0.4, q[0], span / 4, q[1], span / 2],
        [0.6, q[0], span / 2, q[1], span / 4],
    ]
    best = None
    for x0 in inits:
        res = minimize(neg_ll, x0, method="Nelder-Mead",
                       options={"maxiter": 10000, "fatol": 1e-9})
        if best is None or res.fun < best.fun:
            best = res

    w, mu1, s1, mu2, s2 = best.x
    s1, s2 = abs(s1), abs(s2)
    if mu1 > mu2:
        w, mu1, s1, mu2, s2 = 1 - w, mu2, s2, mu1, s1
    return float(w), float(mu1), float(s1), float(mu2), float(s2)


def gmm_pdf(x, w, mu1, s1, mu2, s2) -> np.ndarray:
    return w * sp.norm.pdf(x, mu1, s1) + (1 - w) * sp.norm.pdf(x, mu2, s2)


def gmm_cdf(x, w, mu1, s1, mu2, s2) -> np.ndarray:
    return w * sp.norm.cdf(x, mu1, s1) + (1 - w) * sp.norm.cdf(x, mu2, s2)


_TIME_TICKS = [
    (1,         "1с"),
    (60,        "1мин"),
    (600,       "10мин"),
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
    ax.set_xlabel("Время выполнения")


# ── elapsed.png ───────────────────────────────────────────────────────────────

def plot_elapsed(orig: pd.DataFrame, scale: int = 1000):
    orig_e = orig["ElapsedRaw"].dropna()
    orig_e = orig_e[orig_e > 0].to_numpy()

    fig, ax_top = plt.subplots(figsize=(10, 5))
    fig.suptitle("Распределение времени выполнения  (UID=50109, sphere.slrm)", fontsize=13)

    log_orig  = np.log(orig_e)
    lo_e, hi_e = log_orig.min(), log_orig.max()
    bins_orig  = np.linspace(lo_e, hi_e, 40)
    ctrs_orig  = (bins_orig[:-1] + bins_orig[1:]) / 2
    mu_e       = float(log_orig.mean())

    # Усечённое нормальное в log-пространстве (используется в генераторе)
    mu_tn  = (lo_e + hi_e) / 2
    sig_tn = (hi_e - lo_e) / 6
    a_tn   = (lo_e - mu_tn) / sig_tn
    b_tn   = (hi_e - mu_tn) / sig_tn
    pdf_tn   = sp.truncnorm.pdf(ctrs_orig, a_tn, b_tn, loc=mu_tn, scale=sig_tn)
    ks_tn, _ = sp.kstest(log_orig, sp.truncnorm.cdf, args=(a_tn, b_tn, mu_tn, sig_tn))
    ll_tn    = logloss(log_orig, lambda x: sp.truncnorm.pdf(x, a_tn, b_tn, loc=mu_tn, scale=sig_tn))

    # GMM-2 (только для анализа)
    gw, gmu1, gs1, gmu2, gs2 = fit_gmm2(log_orig)
    pdf_gmm   = gmm_pdf(ctrs_orig, gw, gmu1, gs1, gmu2, gs2)
    ks_gmm, _ = sp.kstest(log_orig, lambda x: gmm_cdf(x, gw, gmu1, gs1, gmu2, gs2))
    ll_gmm    = logloss(log_orig, lambda x: gmm_pdf(x, gw, gmu1, gs1, gmu2, gs2))

    ax_top.hist(log_orig, bins=bins_orig, density=True, alpha=0.6,
                color="steelblue", label=f"Оригинал (n={len(orig_e)})")
    ax_top.plot(ctrs_orig, pdf_tn, "orange", lw=2.5,
                label=f"★ Усечённое норм. (генератор)  KS={ks_tn:.3f}  LL={ll_tn:.3f}")
    ax_top.plot(ctrs_orig, pdf_gmm, "g--", lw=1.5,
                label=f"GMM-2 (анализ)  KS={ks_gmm:.3f}  LL={ll_gmm:.3f}")
    _set_time_ticks(ax_top, lo_e, hi_e)
    ax_top.set_ylabel("Плотность")
    ax_top.set_title(f"Оригинальные данные  медиана={int(np.exp(mu_e))} с ≈ {np.exp(mu_e)/3600:.1f} ч")
    ax_top.legend(fontsize=9)

    plt.tight_layout()
    out = PLOTS_DIR / "elapsed.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Сохранено → {out}")


# ── log_error.png ─────────────────────────────────────────────────────────────

def plot_log_error(orig: pd.DataFrame):
    valid   = orig[(orig["ElapsedRaw"] > 0) & (orig["TimelimitRaw"] > 0)]
    log_err = np.log(valid["TimelimitRaw"] * 60 / valid["ElapsedRaw"])
    log_err = log_err[np.isfinite(log_err)].to_numpy()

    lo, hi = np.quantile(log_err, 0.01), np.quantile(log_err, 0.99)
    data   = log_err.clip(lo, hi)
    bins   = np.linspace(lo, hi, 35)

    norm_params = sp.norm.fit(data)
    ks_n, _     = sp.kstest(data, sp.norm.cdf, args=norm_params)
    ll_n        = logloss(data, lambda x: sp.norm.pdf(x, *norm_params))

    gw, gmu1, gs1, gmu2, gs2 = fit_gmm2(data)
    ks_g, _ = sp.kstest(data, lambda x: gmm_cdf(x, gw, gmu1, gs1, gmu2, gs2))
    ll_g    = logloss(data, lambda x: gmm_pdf(x, gw, gmu1, gs1, gmu2, gs2))

    gmm_label = f"GMM-2  {gw:.2f}·N({gmu1:.1f},{gs1:.1f}) + {1-gw:.2f}·N({gmu2:.1f},{gs2:.1f})"

    fig, ax = plt.subplots(figsize=(11, 6))
    fig.suptitle("Ошибка оценки времени  ln(Timelimit / Elapsed)  (UID=50109, sphere.slrm)",
                 fontsize=13)

    x = np.linspace(lo, hi, 400)
    ax.hist(data, bins=bins, density=True, alpha=0.55, color="steelblue",
            label=f"Данные (n={len(data)})")
    ax.plot(x, sp.norm.pdf(x, *norm_params), "orange", lw=2.5,
            label=f"★ Лог-нормальное (генератор)  KS={ks_n:.3f}  LL={ll_n:.3f}")
    ax.plot(x, gmm_pdf(x, gw, gmu1, gs1, gmu2, gs2), "g--", lw=1.5, alpha=0.85,
            label=f"{gmm_label}  KS={ks_g:.3f}  LL={ll_g:.3f}")

    ax.set_xlabel("ln(Timelimit / Elapsed)")
    ax.set_ylabel("Плотность")
    ax.set_title(f"Ошибка оценки времени  (генератор: KS={ks_n:.3f}, LL={ll_n:.3f} | "
                 f"GMM-2: KS={ks_g:.3f}, LL={ll_g:.3f})")
    ax.legend(fontsize=8)

    plt.tight_layout()
    out = PLOTS_DIR / "log_error.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Сохранено → {out}")


def main():
    args = _parse_args()

    orig = load_orig(args.uid, args.job)
    ref = compute_stats(orig)
    print(ref)

    PLOTS_DIR.mkdir(exist_ok=True)
    plot_elapsed(orig, scale=args.scale)
    plot_log_error(orig)


if __name__ == "__main__":
    main()
