import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import interact, widgets


def plot_sensitivity_lsh():
    # Parameters (tweakable)
    d1, d2 = 2.0, 6.0  # thresholds
    p1, p2 = 0.8, 0.2  # probability guarantees
    d_max = 10

    # Domain
    x = np.linspace(0, d_max, 600)

    # probabilities
    y = np.empty_like(x)
    # define the 3 regions
    left = x <= d1
    mid = (x > d1) & (x < d2)
    right = x >= d2

    # left: quadratic decay from 1 -> p1 (so y(0)=1, y(d1)=p1)
    y[left] = 1.0 - (1.0 - p1) * (x[left] / d1) ** 2

    # mid: linear
    lin = lambda a, x0, x1, y0, y1: y0 + (y1 - y0) * (a - x0) / (x1 - x0)
    y[mid] = lin(x[mid], d1, d2, p1, p2)

    # right: inverse quadratic in a lazy manner
    t = (x[right] - d2) / (d_max - d2)
    y[right] = (1.0 - p2 * (1.0 - t ** 2) - p1)[::-1]

    # Plotting
    plt.figure(figsize=(9, 5))
    plt.plot(x[left], y[left], color="blue", lw=2, label="Example for P(h(A)=h(B))")
    plt.plot(x[mid], y[mid], color="blue", lw=2)
    plt.plot(x[right], y[right], color="blue", lw=2)

    # shading to show the guarantee regions
    plt.fill_between(x, p1, 1.0, where=(x <= d1), alpha=0.18, label=f"P ≥ {p1} when d ≤ {d1}", color="green")
    plt.fill_between(x, 0.0, 1.0, where=(x >= d1) & (x <= d2), alpha=0.18)
    plt.fill_between(x, 0.0, p2, where=(x >= d2), alpha=0.18, label=f"P ≤ {p2} when d ≥ {d2}", color="red")

    # reference lines & vertical thresholds
    plt.axhline(p1, color="green", linestyle="--", linewidth=1)
    plt.axhline(p2, color="red", linestyle="--", linewidth=1)
    plt.axvline(d1, color="gray", linestyle=":", linewidth=1)
    plt.axvline(d2, color="gray", linestyle=":", linewidth=1)

    # markers where guarantees meet the curve
    plt.scatter([d1, d2], [p1, p2], zorder=5)
    plt.text(d1 + 0.1, p1 + 0.03, f"d ≤ {d1}: P ≥ {p1}", color="green")
    plt.text(d2 + 0.1, p2 + 0.03, f"d ≥ {d2}: P ≤ {p2}", color="red")

    # labels
    plt.xlabel("Distance d(A, B)")
    plt.ylabel("Probability of collision P(h(A)=h(B))")
    plt.title("(d1, d2, p1, p2)-sensitive hash family")
    plt.xlim(0, 10)
    plt.ylim(0, 1.05)
    plt.grid(alpha=0.25)
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.show()


def plot_bands_and_rows(r=3, b=5):
    sim = np.arange(0, 1.01, 0.01)
    p = 1 - (1 - sim ** r) ** b

    plt.plot(sim, 1 - (1 - sim ** r) ** b, label="Selected")

    if r != 1:
        plt.plot(sim, 1 - (1 - sim ** (r - 1)) ** b, "-.", label=f"R={r - 1}, B={b}", color="yellow")
    plt.plot(sim, 1 - (1 - sim ** (r + 1)) ** b, "--", label=f"R={r + 1}, B={b}", color="yellow")

    if b != 1:
        plt.plot(sim, 1 - (1 - sim ** (r)) ** (b - 1), "-.", label=f"R={r}, B={b - 1}", color="red")
    plt.plot(sim, 1 - (1 - sim ** (r)) ** (b + 1), "--", label=f"R={r}, B={b + 1}", color="red")

    plt.xlabel("Similarity")
    plt.ylabel("P(candidancy)")
    plt.legend()