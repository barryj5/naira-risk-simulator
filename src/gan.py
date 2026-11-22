from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from data import load_rate
from gan_data import get_gan_data, WINDOW

NOISE_DIM = 16
STEPS = 10000
BATCH = 64
MODEL_PATH = Path("models/gan.pt")

torch.manual_seed(0)
np.random.seed(0)


class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(NOISE_DIM, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, WINDOW),
        )

    def forward(self, z):
        return self.net(z)


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(WINDOW, 64), nn.LeakyReLU(0.2),
            nn.Linear(64, 32), nn.LeakyReLU(0.2),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.net(x)


def train():
    windows, mean, std, last_rate = get_gan_data()
    real_all = torch.tensor(windows)
    G, D = Generator(), Discriminator()
    opt_g = torch.optim.Adam(G.parameters(), lr=2e-4, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(D.parameters(), lr=2e-4, betas=(0.5, 0.999))
    loss_fn = nn.BCEWithLogitsLoss()
    ones, zeros = torch.ones(BATCH, 1), torch.zeros(BATCH, 1)

    for step in range(1, STEPS + 1):
        real = real_all[torch.randint(0, len(real_all), (BATCH,))]
        fake = G(torch.randn(BATCH, NOISE_DIM))

        # 1) teach the discriminator: real -> 1, fake -> 0
        d_loss = loss_fn(D(real), ones) + loss_fn(D(fake.detach()), zeros)
        opt_d.zero_grad()
        d_loss.backward()
        opt_d.step()

        # 2) teach the generator to fool the discriminator
        g_loss = loss_fn(D(fake), ones)
        opt_g.zero_grad()
        g_loss.backward()
        opt_g.step()

        if step % 1000 == 0:
            print(f"step {step}  D loss {d_loss.item():.3f}  G loss {g_loss.item():.3f}")

    MODEL_PATH.parent.mkdir(exist_ok=True)
    torch.save({"G": G.state_dict(), "mean": mean, "std": std,
                "last_rate": last_rate}, MODEL_PATH)


def simulate(n_paths=200, days=WINDOW, model_path=MODEL_PATH):
    days = min(days, WINDOW)
    ckpt = torch.load(model_path)
    G = Generator()
    G.load_state_dict(ckpt["G"])
    G.eval()
    with torch.no_grad():
        scaled = G(torch.randn(n_paths, NOISE_DIM)).numpy()
    scaled = scaled - scaled.mean(axis=0, keepdims=True)   # remove the GAN's built-in drift
    returns = scaled[:, :days] * ckpt["std"] + ckpt["mean"]   # back to real % changes
    paths = ckpt["last_rate"] * np.cumprod(1 + returns, axis=1)   # back to naira rates
    return paths, returns


def bootstrap_paths(n_paths=200, days=WINDOW):
    """Baseline: build paths by reshuffling real daily changes at random."""
    rate = load_rate()
    returns = rate.pct_change().dropna().to_numpy()
    rng = np.random.default_rng(0)
    sampled = rng.choice(returns, size=(n_paths, days), replace=True)
    return rate.iloc[-1] * np.cumprod(1 + sampled, axis=1), sampled


if __name__ == "__main__":
    train()
    paths, gen_returns = simulate(200)
    boot_paths, boot_returns = bootstrap_paths(200)

    windows, mean, std, last_rate = get_gan_data()
    real_returns = windows * std + mean

    def cum(r):
        return np.prod(1 + r, axis=1) - 1

    print(f"{WINDOW}-day change (mean %, std %):")
    for name, r in [("REAL", real_returns), ("GAN", gen_returns), ("BOOTSTRAP", boot_returns)]:
        print("  %-10s %6.2f  %6.2f" % (name, cum(r).mean() * 100, cum(r).std() * 100))
    print("Per-day mean of GAN returns (%):", np.round(gen_returns.mean(axis=0) * 100, 2))
    print("Per-day std of GAN returns (%): ", np.round(gen_returns.std(axis=0) * 100, 2))
    print("Real daily std (%):", round(std * 100, 2))

    rate = load_rate()
    plt.plot(range(-29, 1), rate.iloc[-30:].to_numpy(), color="black", label="history")
    for p in paths[:50]:
        plt.plot(range(0, WINDOW + 1), np.concatenate([[last_rate], p]), alpha=0.25)
    plt.title("GAN simulated paths (50 of 200)")
    plt.xlabel("business days from last known rate")
    plt.ylabel("naira rate")
    plt.legend()
    plt.show()