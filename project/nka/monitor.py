"""Monitoring helpers: drift (PSI), query log, synthetic traffic, A/B test."""
import csv, time
from math import sqrt
import numpy as np, pandas as pd
from scipy import stats
from .config import ARTIFACTS

LOG_FILE = ARTIFACTS / "query_log.csv"
LOG_COLS = ["timestamp", "question", "answer", "top_score", "latency_ms", "feedback"]


def psi(reference, current, bins: int = 10) -> float:
    """Population Stability Index. Rule of thumb: <0.1 stable, 0.1-0.25 watch, >0.25 drifted."""
    ref, cur = np.asarray(reference, float), np.asarray(current, float)
    edges = np.quantile(ref, np.linspace(0, 1, bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    edges = np.unique(edges)
    p = np.histogram(ref, edges)[0] / len(ref) + 1e-6
    q = np.histogram(cur, edges)[0] / len(cur) + 1e-6
    return float(np.sum((q - p) * np.log(q / p)))


def log_query(question, answer, top_score, latency_ms, feedback=None):
    """Append one row to artifacts/query_log.csv."""
    new = not LOG_FILE.exists()
    with open(LOG_FILE, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new: w.writerow(LOG_COLS)
        w.writerow([time.strftime("%Y-%m-%d %H:%M:%S"), question, str(answer)[:300], round(float(top_score), 3),
                    round(float(latency_ms), 1), "" if feedback is None else feedback])


def simulate_traffic(n: int = 500, drift: bool = False, seed: int = 0) -> pd.DataFrame:
    """Synthetic query features. drift=True: longer queries, lower retrieval scores, slower, fewer thumbs up."""
    rng = np.random.default_rng(seed)
    length = np.clip(rng.normal(9 + (6 if drift else 0), 3, n), 2, None).round()
    score = np.clip(rng.normal(0.72 - (0.12 if drift else 0), 0.08, n), 0.2, 0.95)
    latency = np.clip(rng.lognormal(6.0 + (0.15 if drift else 0), 0.3, n), 50, None)
    thumbs = (rng.random(n) < np.clip(0.4 + 1.2 * (score - 0.5), 0.05, 0.95)).astype(int)
    return pd.DataFrame(dict(query_length=length, top_score=score.round(3), latency_ms=latency.round(1), thumbs_up=thumbs))


def ab_test(success_a, n_a, success_b, n_b, alpha: float = 0.05) -> dict:
    """Two-proportion z-test: lift of B over A, z, two-sided p, and a decision."""
    pa, pb = success_a / n_a, success_b / n_b
    pool = (success_a + success_b) / (n_a + n_b)
    se = sqrt(pool * (1 - pool) * (1 / n_a + 1 / n_b))
    z = (pb - pa) / se if se else 0.0
    p = float(2 * (1 - stats.norm.cdf(abs(z))))
    decision = "ship B" if p < alpha and pb > pa else "keep A" if p < alpha else "no significant difference: keep collecting data"
    return dict(rate_a=round(pa, 4), rate_b=round(pb, 4), lift=round(pb - pa, 4), z=round(z, 3), p=round(p, 4), decision=decision)
