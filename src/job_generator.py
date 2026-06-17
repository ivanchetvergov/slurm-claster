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

    def __init__(self, stats: JobStats, seed: int | None = None, time_scale: int = 1, max_seconds: int = 3600):
        self.stats = stats
        self.rng = np.random.default_rng(seed)
        self.time_scale = max(1, time_scale)
        self.max_seconds = max_seconds

    def generate(self, n: int) -> list[JobRequest]:
        if not 1 <= n <= 999:
            raise ValueError(f"n должно быть от 1 до 999, получено {n}")
        if self.stats.elapsed_min <= 0:
            raise ValueError(f"elapsed_min должен быть > 0, получено {self.stats.elapsed_min}")

        mu_e    = (self.stats.elapsed_min + self.stats.elapsed_max) / 2
        sigma_e = (self.stats.elapsed_max - self.stats.elapsed_min) / 6
        a = (self.stats.elapsed_min - mu_e) / sigma_e
        b = (self.stats.elapsed_max - mu_e) / sigma_e
        elapsed = truncnorm.rvs(a, b, loc=mu_e, scale=sigma_e,
                                size=n, random_state=self.rng)

        log_error = self.rng.normal(self.stats.log_error_mu, self.stats.log_error_sigma, size=n)
        log_error = np.clip(log_error, None, self.stats.log_error_clip)
        timelimit = elapsed * np.exp(log_error)
        nodes     = self.rng.integers(1, 5, size=n)

        elapsed_s   = np.maximum(1, np.round(elapsed  / self.time_scale).astype(int))
        timelimit_s = np.maximum(elapsed_s, np.round(timelimit / self.time_scale).astype(int))
        timelimit_s = np.minimum(timelimit_s, self.max_seconds)

        return [JobRequest(int(e), int(t), int(nd))
                for e, t, nd in zip(elapsed_s, timelimit_s, nodes)]
