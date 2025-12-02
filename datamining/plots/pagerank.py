import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import HTML


def random_walk_transition_matrix(A):
    """Given an adjacency matrix A, construct the random-walk transition matrix P.
    If a row has all zeros (sink node), we keep the row as all zeros here
    """
    A = np.array(A, dtype=float)
    n = A.shape[0]
    P = np.zeros_like(A)
    for i in range(n):
        outdeg = A[i].sum()
        if outdeg > 0:
            P[i] = A[i] / outdeg
        else:
            P[i] = 0  # sink
    return P


def pagerank_history(A, alpha=0.85, v=None, max_iter=10):
    """
    Run PageRank power iteration and return list of pi^(t) for t = 0..T.
    """
    A = np.array(A, dtype=float)
    n = A.shape[0]
    P = random_walk_transition_matrix(A)

    # personalization / teleportation vector
    if v is None:
        v = np.ones(n)
    else:
        v = np.array(v, dtype=float)
    v = v / v.sum()

    # handle sinks: replace zero rows with v
    for i in range(n):
        if np.allclose(P[i], 0):
            P[i] = v

    # initial distribution (can start from v or uniform)
    pi = np.ones(n) / n

    history = [pi.copy()]
    for _ in range(max_iter):
        pi = alpha * (pi @ P) + (1 - alpha) * v
        history.append(pi.copy())
    return np.array(history)   # shape: (T+1, n)


def plot_pagerank_iteration(A, alpha=0.85, v=None, max_iter=10):
    history = pagerank_history(A, alpha=alpha, v=v, max_iter=max_iter)

    num_steps, n = history.shape

    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.arange(n)
    bars = ax.bar(x, history[0])

    ax.set_ylim(0, history.max() * 1.1)
    ax.set_xticks(x)
    ax.set_xticklabels([f"Node {i}" for i in range(n)])
    ax.set_ylabel("PageRank value")
    title = ax.set_title("PageRank iteration t = 0")

    def update(frame):
        y = history[frame]
        for bar, h in zip(bars, y):
            bar.set_height(h)
        title.set_text(f"PageRank iteration t = {frame}")
        return bars, title

    anim = FuncAnimation(fig, update, frames=num_steps, interval=400, blit=False)

    plt.close(fig)  # prevent double display
    return HTML(anim.to_jshtml())