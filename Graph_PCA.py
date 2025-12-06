import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np
from collections import Counter
from itertools import combinations
from mpl_toolkits.mplot3d import Axes3D  # nötig für 3D-Projektion


def plot_clusters_pca(
        X,
        labels,
        kmeans_model=None,
        title="Clusters in PCA space",
        random_state=42,
        n_components=3,
        pairs=None,
        plot_3d=False,
        components_3d=(0, 1, 2),
):
    

    # PCA mit mehr als 2 Komponenten
    reducer = PCA(n_components=n_components, random_state=random_state)
    X_pca = reducer.fit_transform(X)

    if pairs is None:
        pairs = [(0, 1)]

    # Sicherstellen, dass die gewünschten 2D-Komponenten existieren
    max_comp_index_2d = max(max(i, j) for i, j in pairs)
    # Sicherstellen, dass die gewünschten 3D-Komponenten existieren
    max_comp_index_3d = max(components_3d) if plot_3d else -1

    max_needed = max(max_comp_index_2d, max_comp_index_3d)
    if max_needed >= n_components:
        raise ValueError(
            f"Es werden Komponenten bis {max_needed} benötigt, "
            f"aber n_components={n_components}. Bitte n_components erhöhen."
        )

    # Clustergrößen
    counts = Counter(labels)

    # Farbenpalette
    colors = [
        "#1f77b4", "#2ca02c", "#ff7f0e", "#d62728", "#9467bd",
        "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"
    ]

    # Zentren in PCA-Raum projizieren
    centers_pca = None
    if kmeans_model is not None:
        centers_pca = reducer.transform(kmeans_model.cluster_centers_)

    # ---------- 2D: Für jedes Komponentenpaar eine eigene Figur zeichnen ----------
    for (i, j) in pairs:
        plt.figure(figsize=(10, 8))

        for cl in sorted(set(labels)):
            idxs = np.where(labels == cl)[0]
            plt.scatter(
                X_pca[idxs, i],
                X_pca[idxs, j],
                c=colors[cl % len(colors)],
                s=30,
                alpha=0.7,
                label=f"Cluster {cl} (n={counts[cl]})"
            )

        # Zentren plotten (falls vorhanden)
        if centers_pca is not None:
            plt.scatter(
                centers_pca[:, i],
                centers_pca[:, j],
                c="black",
                s=200,
                alpha=0.9,
                marker="X",
                label="Cluster centers"
            )

        plt.title(f"{title}: PC{i + 1} vs. PC{j + 1}")
        plt.xlabel(f"Principal component {i + 1}")
        plt.ylabel(f"Principal component {j + 1}")
        plt.legend()
        plt.tight_layout()
        plt.show()

    # ---------- 3D: optionaler 3D-Plot ----------
    if plot_3d:
        a, b, c = components_3d  # z.B. 0,1,2 für PC1, PC2, PC3

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        for cl in sorted(set(labels)):
            idxs = np.where(labels == cl)[0]
            ax.scatter(
                X_pca[idxs, a],
                X_pca[idxs, b],
                X_pca[idxs, c],
                c=colors[cl % len(colors)],
                s=30,
                alpha=0.7,
                label=f"Cluster {cl} (n={counts[cl]})"
            )

        if centers_pca is not None:
            ax.scatter(
                centers_pca[:, a],
                centers_pca[:, b],
                centers_pca[:, c],
                c="black",
                s=200,
                alpha=0.9,
                marker="X",
                label="Cluster centers"
            )

        ax.set_title(f"{title} (3D): PC{a + 1} vs. PC{b + 1} vs. PC{c + 1}")
        ax.set_xlabel(f"Principal component {a + 1}")
        ax.set_ylabel(f"Principal component {b + 1}")
        ax.set_zlabel(f"Principal component {c + 1}")
        ax.legend()
        plt.tight_layout()
        plt.show()
