"""K-Means clustering analysis on a dataset.

Plots the elbow curve (k=1..11) to suggest k, then fits KMeans at a fixed k per
dataset, shows a 3D cluster plot (top 3 features by ANOVA), and prints
silhouette + adjusted Rand index against the true labels.
"""

import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
import numpy as np
from sklearn import cluster, metrics
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import StandardScaler
from yellowbrick.cluster import KElbowVisualizer

from data_science.datasets import DATASETS

# Per-dataset choice of k for the final KMeans fit (from the team's elbow analysis).
_K_BY_SOURCE = {"PD": 4, "CT": 3}


def run(data, source):
    target = DATASETS[source].target_column
    y = data.pop(target).values.astype(int)
    X = StandardScaler().fit_transform(data.values)

    # 1. Elbow plot to suggest k
    visualizer = KElbowVisualizer(cluster.KMeans(random_state=1, n_init=10), k=(1, 12))
    visualizer.fit(X)
    visualizer.show()

    # 2. Fit KMeans at the chosen k for this dataset
    n_clusters = _K_BY_SOURCE[source]
    kmeans = cluster.KMeans(n_clusters=n_clusters, random_state=1, n_init=10).fit(X)
    y_pred = kmeans.labels_

    # 3. 3D scatter on the top 3 features by ANOVA against the true labels
    X_3d = SelectKBest(f_classif, k=3).fit_transform(X, y)
    fig = plt.figure()
    ax = p3.Axes3D(fig)
    ax.view_init(7, -80)
    for label in np.unique(y_pred):
        mask = y_pred == label
        ax.scatter(
            X_3d[mask, 0],
            X_3d[mask, 1],
            X_3d[mask, 2],
            color=plt.cm.jet(float(label) / np.max(y_pred + 1)),
            s=20,
            edgecolor="k",
        )
    plt.title("Clustering Solution")
    plt.show()

    # 4. Metrics
    print("Clustering:")
    print(f"  Number of clusters:             {n_clusters}")
    print(f"  Sum of squared distances:       {kmeans.inertia_:.2f}")
    print(f"  Average silhouette coefficient: {metrics.silhouette_score(X, y_pred):.4f}")
    print(f"  Adjusted Rand index:            {adjusted_rand_score(y, y_pred):.4f}")
