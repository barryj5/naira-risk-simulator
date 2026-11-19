import numpy as np
from data import load_rate

WINDOW = 20   # each training example is 20 business days long


def make_windows(returns, window=WINDOW):
    values = returns.to_numpy(dtype="float32")
    return np.stack([values[i:i + window] for i in range(len(values) - window + 1)])


def get_gan_data():
    rate = load_rate()
    returns = rate.pct_change().dropna()      # daily % change
    mean, std = returns.mean(), returns.std()
    scaled = (returns - mean) / std           # centre around 0, spread of 1
    windows = make_windows(scaled)
    return windows, float(mean), float(std), float(rate.iloc[-1])


if __name__ == "__main__":
    windows, mean, std, last_rate = get_gan_data()
    print("Windows shape:", windows.shape)
    print("Mean daily change:", mean, "Std:", std)
    print("Last known rate:", last_rate)