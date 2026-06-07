from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay


def plot_confusion_matrix(
    cm: list[list[int]] | np.ndarray,
    class_names: list[str],
    output_path: str | Path,
    title: str = "Confusion matrix",
) -> None:
    """Save a confusion matrix as a PNG.

    cm: 2D array or nested list of counts.
    class_names: labels for both axes (length must equal the matrix side).
    output_path: file path to write the PNG.
    """
    cm_array = np.asarray(cm)
    fig, ax = plt.subplots(figsize=(max(6, len(class_names)), max(5, len(class_names) - 1)))
    display = ConfusionMatrixDisplay(
        confusion_matrix=cm_array,
        display_labels=class_names,
    )
    display.plot(ax=ax, cmap="Blues", values_format="d", colorbar=False)
    ax.set_title(title)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    plt.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
