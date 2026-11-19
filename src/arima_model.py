import itertools
import warnings
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from data import load_rate

warnings.filterwarnings("ignore")

rate = load_rate()
test_size = 20
train, test = rate[:-test_size], rate[-test_size:]

# Try several (p, 1, q) settings and rank them by AIC (lower is better)
results = []
for p, q in itertools.product(range(4), range(4)):
    try:
        m = ARIMA(train, order=(p, 1, q)).fit()
        results.append((p, 1, q, m.aic))
    except Exception:
        pass
best = sorted(results, key=lambda r: r[3])[:5]
print("Top 5 by AIC:", best)

order = best[0][:3]
fit = ARIMA(train, order=order).fit()
fc = fit.get_forecast(steps=test_size)
pred = fc.predicted_mean
ci = fc.conf_int()

naive = pd.Series(train.iloc[-1], index=test.index)
print("Chosen order:", order)
print("ARIMA MAE:", (test - pred).abs().mean())
print("Naive MAE:", (test - naive).abs().mean())

train[-60:].plot(label="train")
test.plot(label="actual")
pred.plot(label="ARIMA forecast")
plt.fill_between(ci.index, ci.iloc[:, 0], ci.iloc[:, 1], alpha=0.2)
plt.legend()
plt.show()