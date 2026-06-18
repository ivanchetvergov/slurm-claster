import numpy as np
from dataclasses import dataclass
from scipy.stats import truncnorm
from stats_calculator import JobStats, LOG_ERROR_CLIP, LOG_ERROR_LO


@dataclass
class JobRequest:
    elapsed_sec: float
    timelimit_sec: int
    nodes: int
    log_error: float


class JobGenerator:

    def __init__(self, stats: JobStats, seed: int | None = None, time_scale: int = 1, max_seconds: int = 3600):
        self.stats = stats
        self.rng = np.random.default_rng(seed)
        self.time_scale = max(1, time_scale)
        self.max_seconds = max_seconds

    def _sample_elapsed(self, n: int) -> np.ndarray:
        s = self.stats
        component = self.rng.choice(2, size=n, p=[s.elapsed_gmm_w, 1 - s.elapsed_gmm_w])
        log_e = np.where(
            component == 0,
            self.rng.normal(s.elapsed_gmm_mu1, s.elapsed_gmm_sig1, n),
            self.rng.normal(s.elapsed_gmm_mu2, s.elapsed_gmm_sig2, n),
        )
        return np.clip(np.exp(log_e), s.elapsed_min, s.elapsed_max)

    def _sample_log_error(self, n: int) -> np.ndarray:
        s = self.stats
        a = (LOG_ERROR_LO  - s.log_error_gmm1_mu) / s.log_error_gmm1_sig
        b = (LOG_ERROR_CLIP - s.log_error_gmm1_mu) / s.log_error_gmm1_sig
        return truncnorm.rvs(a, b, loc=s.log_error_gmm1_mu, scale=s.log_error_gmm1_sig,
                             size=n, random_state=self.rng)

    def generate(self, n: int) -> list[JobRequest]:
        if not 1 <= n <= 999:
            raise ValueError(f"n должно быть от 1 до 999, получено {n}")

        elapsed   = self._sample_elapsed(n)
        log_error = self._sample_log_error(n)
        timelimit = elapsed * np.exp(log_error)
        nodes     = self.rng.integers(1, 5, size=n)

        elapsed_s   = np.maximum(0.01, np.round(elapsed  / self.time_scale, 2))
        elapsed_s   = np.minimum(elapsed_s, self.max_seconds)
        timelimit_s = np.maximum(1, np.round(timelimit / self.time_scale).astype(int))
        timelimit_s = np.minimum(timelimit_s, self.max_seconds)

        return [JobRequest(float(e), int(t), int(nd), float(le))
                for e, t, nd, le in zip(elapsed_s, timelimit_s, nodes, log_error)]
