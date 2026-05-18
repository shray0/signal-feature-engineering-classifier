"""
Visualization helpers for 187 x 8 signal samples.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np


def plot_signal_sample(sample: np.ndarray, label: str | None = None) -> None:
    """
    Plot all 8 channels from one 187 x 8 sample.
    """
    plt.figure(figsize=(12, 6))
    for channel in range(sample.shape[1]):
        plt.plot(sample[:, channel], alpha=0.35 + 0.06 * channel, label=f"Channel {channel}")

    title = "Signal sample"
    if label is not None:
        title += f" ({label})"

    plt.title(title)
    plt.xlabel("Time index")
    plt.ylabel("Value")
    plt.grid(True)
    plt.legend(ncol=4, fontsize=8)
    plt.tight_layout()
    plt.show()
