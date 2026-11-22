from functools import lru_cache
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from data import load_rate

ORDER = (2, 1, 3)   # chosen by AIC in arima_model.py


@lru_cache(maxsize=1)
def _fit():
    rate = load_rate()
    return rate, ARIMA(rate, order=ORDER).fit()


def arima_forecast(days=7):
    rate, fit = _fit()
    fc = fit.get_forecast(steps=days)
    mean = fc.predicted_mean
    ci = fc.conf_int()
    return {
        "last_date": str(rate.index[-1].date()),
        "last_rate": float(rate.iloc[-1]),
        "dates": [str(d.date()) for d in mean.index],
        "forecast": [round(float(v), 2) for v in mean],
        "lower": [round(float(v), 2) for v in ci.iloc[:, 0]],
        "upper": [round(float(v), 2) for v in ci.iloc[:, 1]],
    }


def simulate_paths(method="bootstrap", n_paths=300, days=10):
    days = max(1, min(days, 10))
    rate = load_rate()
    if method == "gan":
        from gan import simulate
        paths, _ = simulate(n_paths, days)
    else:
        returns = rate.pct_change().dropna().to_numpy()
        rng = np.random.default_rng(0)
        sampled = rng.choice(returns, size=(n_paths, days), replace=True)
        paths = float(rate.iloc[-1]) * np.cumprod(1 + sampled, axis=1)
    return {
        "method": method,
        "last_rate": float(rate.iloc[-1]),
        "p5": np.percentile(paths, 5, axis=0).round(2).tolist(),
        "p50": np.percentile(paths, 50, axis=0).round(2).tolist(),
        "p95": np.percentile(paths, 95, axis=0).round(2).tolist(),
        "sample_paths": paths[:20].round(2).tolist(),
    }