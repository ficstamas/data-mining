import numpy as np
import matplotlib.pyplot as plt


def plot_lda(n_classes=3):
    np.random.seed(0)
    n_samples_per_class = 20
    means = np.array([[0, 0], [3, 2], [0, 4], [-2, 1], [-5, 5]])[:n_classes]  # class means (rows = classes)
    cov = np.array([[1.0, 0.5], [0.5, 1.0]])  # shared covariance matrix
    colors = ['red', 'green', 'blue', 'gray', 'orange'][:n_classes]  # class colors
    scale_factor = 4.0  # for drawing the LDA line
    # ================================================================

    # ------------------------------------------------
    # Generate synthetic data
    # ------------------------------------------------
    X = []
    y = []
    for i in range(n_classes):
        X_i = np.random.multivariate_normal(means[i], cov, n_samples_per_class)
        X.append(X_i)
        y += [i] * n_samples_per_class
    X = np.vstack(X)
    y = np.array(y)

    # ------------------------------------------------
    # LDA implementation (1D)
    # ------------------------------------------------
    class LDA:
        def fit(self, X, y):
            n_features = X.shape[1]
            classes = np.unique(y)
            mean_overall = np.mean(X, axis=0)
            Sw = np.zeros((n_features, n_features))
            Sb = np.zeros((n_features, n_features))

            for c in classes:
                X_c = X[y == c]
                mean_c = np.mean(X_c, axis=0)
                Sw += (X_c - mean_c).T @ (X_c - mean_c)
                n_c = X_c.shape[0]
                mean_diff = (mean_c - mean_overall).reshape(n_features, 1)
                Sb += n_c * mean_diff @ mean_diff.T

            eigvals, eigvecs = np.linalg.eig(np.linalg.pinv(Sw) @ Sb)
            w = eigvecs[:, np.argmax(eigvals)].real
            self.w = w / np.linalg.norm(w)

        def transform(self, X):
            return X @ self.w

    # ------------------------------------------------
    # Fit LDA and project
    # ------------------------------------------------
    lda = LDA()
    lda.fit(X, y)
    X_proj = lda.transform(X)

    # Points projected back into 2D space for visualization
    mean_all = np.mean(X, axis=0)
    proj_points = np.outer(X_proj, lda.w)
    proj_points += mean_all - np.mean(proj_points, axis=0)

    # ------------------------------------------------
    # Plot
    # ------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 6))

    # Original data
    for i, color in enumerate(colors[:n_classes]):
        ax.scatter(X[y == i, 0], X[y == i, 1], label=f"Class {i}", alpha=0.7, color=color)

    # LDA direction line
    lda_line = np.vstack([
        mean_all - scale_factor * lda.w,
        mean_all + scale_factor * lda.w
    ])
    ax.plot(lda_line[:, 0], lda_line[:, 1], 'k--', linewidth=2, label='LDA direction')

    # Projected points (optional visualization)
    for i, color in enumerate(colors[:n_classes]):
        ax.scatter(proj_points[y == i, 0], proj_points[y == i, 1], color=color, marker='x', label='Projections')

    ax.set_title("LDA Demo with Random 2D Data")
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.show()