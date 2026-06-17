import numpy as np
from dataclasses import dataclass
from scipy.stats import truncnorm
from stats_calculator import JobStats


@dataclass
class JobRequest:
    elapsed_sec: int
    timelimit_sec: int
    nodes: int


class JobGenerator:

    def __init__(self, stats: JobStats, seed: int | None = None, time_scale: int = 1):
        self.stats = stats
        self.rng = np.random.default_rng(seed)
        self.time_scale = max(1, time_scale)

    def generate(self, n: int) -> list[JobRequest]:
        if not 1 <= n <= 999:
            raise ValueError(f"n должно быть от 1 до 999, получено {n}")
        if self.stats.elapsed_min <= 0:
            raise ValueError(f"elapsed_min должен быть > 0, получено {self.stats.elapsed_min}")

        ln_lo    = np.log(self.stats.elapsed_min)
        ln_hi    = np.log(self.stats.elapsed_max)
        mu_ln    = (ln_lo + ln_hi) / 2
        sigma_ln = (ln_hi - ln_lo) / 6
        # a = (ln_lo - mu_ln) / sigma_ln = -3, b = +3 by construction
        log_elapsed = truncnorm.rvs(-3.0, 3.0, loc=mu_ln, scale=sigma_ln,
                                    size=n, random_state=self.rng)
        elapsed     = np.exp(log_elapsed)

        log_error = self.rng.normal(self.stats.log_error_mu, self.stats.log_error_sigma, size=n)
        timelimit = elapsed * np.exp(log_error)
        nodes     = self.rng.integers(1, 5, size=n)

        elapsed_s  = np.maximum(1, np.round(elapsed  / self.time_scale).astype(int))
        timelimit_s = np.maximum(elapsed_s, np.round(timelimit / self.time_scale).astype(int))

        return [JobRequest(int(e), int(t), int(nd))
                for e, t, nd in zip(elapsed_s, timelimit_s, nodes)]
