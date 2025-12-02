import numpy as np
from .pagerank import random_walk_transition_matrix
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import HTML


def global_conv_history(A, max_iter):
    P_rw = random_walk_transition_matrix(A)

    # Global convergence
    M = P_rw
    # Fix random seed
    np.random.seed(42)

    # Generate 6 random row vectors from [0,1]
    P = np.random.rand(6, 5)

    # Normalize rows to become probability distributions
    P = P / P.sum(axis=1, keepdims=True)

    # Power iteration
    P_hist = [P]
    for k in range(max_iter):
        P = P @ M
        P_hist.append(P)

    return np.array(P_hist)


def plot_global_conv_iteration(A, max_iter=100):
    history = global_conv_history(A, max_iter=max_iter)

    num_steps, n = history.shape

    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.arange(n)
    bars = ax.bar(x, history[0])

    ax.set_ylim(0, history.max() * 1.1)
    ax.set_xticks(x)
    ax.set_xticklabels([f"Node {i}" for i in range(n)])
    ax.set_ylabel("Stationary value")
    title = ax.set_title("Global convergence iteration t = 0")

    def update(frame):
        y = history[frame]
        for bar, h in zip(bars, y):
            bar.set_height(h)
        title.set_text(f"Global convergence iteration t = {frame}")
        return bars, title

    anim = FuncAnimation(fig, update, frames=num_steps, interval=400, blit=False)

    plt.close(fig)  # prevent double display
    return HTML(anim.to_jshtml())
