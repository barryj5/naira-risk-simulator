import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from data import load_rate

rate = load_rate()

print(rate.describe())
print("Rows:", len(rate))
print("From:", rate.index.min(), "To:", rate.index.max())
print("Rate p-value:", adfuller(rate)[1])
print("Differenced p-value:", adfuller(rate.diff().dropna())[1])

rate.plot(title="NFEM closing rate")
plt.show()