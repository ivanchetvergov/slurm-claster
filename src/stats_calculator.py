import numpy as np
import pandas as pd
from dataclasses import dataclass
from scipy import stats as sp
from scipy.optimize import minimize


LOG_ERROR_CLIP = 4.5
LOG_ERROR_LO   = -2.0


def fit_gmm2(data: np.ndarray):
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


@dataclass
class JobStats:
    elapsed_min: float
    elapsed_max: float
    elapsed_mean: float
    elapsed_std: float
    elapsed_gmm_w:    float  # вес первой компоненты GMM-2 (в log-масштабе)
    elapsed_gmm_mu1:  float
    elapsed_gmm_sig1: float
    elapsed_gmm_mu2:  float
    elapsed_gmm_sig2: float
    log_error_mu: float
    log_error_sigma: float
    log_error_clip: float
    log_error_gmm_w:    float   # вес первой компоненты
    log_error_gmm1_mu:  float
    log_error_gmm1_sig: float
    log_error_gmm2_mu:  float
    log_error_gmm2_sig: float
    sample_size: int

    def __str__(self):
        return (
            f"  elapsed:  min={self.elapsed_min:.0f}s  max={self.elapsed_max:.0f}s"
            f"  mean={self.elapsed_mean:.0f}s  std={self.elapsed_std:.0f}s\n"
            f"  elapsed GMM-2: w={self.elapsed_gmm_w:.2f}"
            f"  N({self.elapsed_gmm_mu1:.2f},{self.elapsed_gmm_sig1:.2f})"
            f"  N({self.elapsed_gmm_mu2:.2f},{self.elapsed_gmm_sig2:.2f})\n"
            f"  log_error GMM-2: {self.log_error_gmm_w:.2f}·N({self.log_error_gmm1_mu:.2f},{self.log_error_gmm1_sig:.2f})"
            f" + {1-self.log_error_gmm_w:.2f}·N({self.log_error_gmm2_mu:.2f},{self.log_error_gmm2_sig:.2f})\n"
            f"  записей: {self.sample_size}"
        )


def compute_stats(df: pd.DataFrame) -> JobStats:
    elapsed = pd.to_numeric(df["ElapsedRaw"], errors="coerce")
    timelimit_sec = pd.to_numeric(df["TimelimitRaw"], errors="coerce") * 60

    in_sample = df["State"].isin(["COMPLETED", "TIMEOUT"])
    e_all = elapsed[in_sample & (elapsed > 0)].to_numpy()

    log_e = np.log(e_all)
    gmm_w, gmm_mu1, gmm_sig1, gmm_mu2, gmm_sig2 = fit_gmm2(log_e)

    completed = df["State"] == "COMPLETED"
    e_c = elapsed[completed]
    t_c = timelimit_sec[completed]
    valid = (e_c > 0) & (t_c > 0)
    log_errors_full = np.log(t_c[valid] / e_c[valid])
    log_errors_full = log_errors_full[np.isfinite(log_errors_full)].to_numpy()
    log_errors = log_errors_full[log_errors_full < LOG_ERROR_CLIP]

    le_w, le_mu1, le_sig1, le_mu2, le_sig2 = fit_gmm2(log_errors_full)

    return JobStats(
        elapsed_min=float(e_all.min()),
        elapsed_max=float(e_all.max()),
        elapsed_mean=float(e_all.mean()),
        elapsed_std=float(e_all.std(ddof=1)),
        elapsed_gmm_w=gmm_w,
        elapsed_gmm_mu1=gmm_mu1,
        elapsed_gmm_sig1=gmm_sig1,
        elapsed_gmm_mu2=gmm_mu2,
        elapsed_gmm_sig2=gmm_sig2,
        log_error_mu=float(log_errors.mean()),
        log_error_sigma=float(log_errors.std(ddof=1)),
        log_error_clip=LOG_ERROR_CLIP,
        log_error_gmm_w=le_w,
        log_error_gmm1_mu=le_mu1,
        log_error_gmm1_sig=le_sig1,
        log_error_gmm2_mu=le_mu2,
        log_error_gmm2_sig=le_sig2,
        sample_size=len(df),
    )
