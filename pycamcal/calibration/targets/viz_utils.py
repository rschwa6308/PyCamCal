from typing import Literal

import numpy as np
from matplotlib import pyplot as plt
import matplotlib.patheffects as pe



def plot_target_detection(image: np.ndarray, target_points: np.ndarray, marker_type: Literal["x", "o"] = "x", zoom: bool = False, zoom_pad: int = 50, marker_size=None, select_keys=None, annotate=False, select_keys_annotate=None, legend=True, ax=None):
    "Plot the given image annotated with the given target points"

    # downselect keys
    if select_keys is None:
        select_keys = target_points.keys()

    # convert to numpy array for convenient indexing
    points = np.array([
        target_points[key] for key in sorted(select_keys)
    ]).reshape((-1, 2))

    if ax is None:
        fig, ax = plt.subplots()

    if marker_type == "x":
        marker_kwargs = dict(marker="x", s=marker_size)
    elif marker_type == "o":
        marker_kwargs = dict(facecolors="none", marker="o", s=marker_size) # open circle

    ax.imshow(image)
    ax.scatter(points[:,0], points[:,1], color="magenta", **marker_kwargs, label=f"target detections ({len(points)})")

    if len(points) == 0: return ax

    if annotate:
        if select_keys_annotate is None:
            select_keys_annotate = select_keys

        for key in select_keys_annotate:
            label = str(key)
            pos = target_points[key]
            ax.annotate(
                label, pos,
                color="magenta",
                xytext=(6, 6),
                textcoords="offset points",
                fontsize=10,
                # fontweight="bold",
                fontfamily="monospace",
                bbox=dict(
                    facecolor="white",
                    edgecolor="none",
                    alpha=0.9,
                    pad=0.25,
                    boxstyle="round"
                ),
            )

    if zoom:
        xmin = np.min(points[:,0])
        xmax = np.max(points[:,0])
        ymin = np.min(points[:,1])
        ymax = np.max(points[:,1])

        ax.set_xlim(xmin-zoom_pad, xmax+zoom_pad)
        ax.set_ylim(ymax+zoom_pad, ymin-zoom_pad)

    if legend:
        ax.legend()

    return ax


# def plot_target_detections
