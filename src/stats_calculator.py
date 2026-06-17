import numpy as np
import pandas as pd
from dataclasses import dataclass


@dataclass
class JobStats:
    elapsed_min: float
    elapsed_max: float
    elapsed_mean: float
    elapsed_std: float
    log_error_mu: float
    log_error_sigma: float
    sample_size: int

    def __str__(self):
        return (
            f"  elapsed:  min={self.elapsed_min:.0f}s  max={self.elapsed_max:.0f}s"
            f"  mean={self.elapsed_mean:.0f}s  std={self.elapsed_std:.0f}s\n"
            f"  log_error: mu={self.log_error_mu:.4f}  sigma={self.log_error_sigma:.4f}\n"
            f"  записей: {self.sample_size}"
        )


class StatsCalculator:

    def __init__(self, df: pd.DataFrame):
        self.df = df

    def compute(self) -> JobStats:
        elapsed = pd.to_numeric(self.df["ElapsedRaw"], errors="coerce")
        timelimit_sec = pd.to_numeric(self.df["TimelimitRaw"], errors="coerce") * 60

        valid = (elapsed > 0) & (timelimit_sec > 0)
        e_valid = elapsed[valid]
        log_errors = np.log(timelimit_sec[valid] / e_valid)
        log_errors = log_errors[np.isfinite(log_errors)]

        return JobStats(
            elapsed_min=float(e_valid.min()),
            elapsed_max=float(e_valid.max()),
            elapsed_mean=float(e_valid.mean()),
            elapsed_std=float(e_valid.std(ddof=1)),
            log_error_mu=float(log_errors.mean()),
            log_error_sigma=float(log_errors.std(ddof=1)),
            sample_size=len(self.df),
        )
