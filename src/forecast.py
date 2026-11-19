from functools import lru_cache
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