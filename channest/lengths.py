import shapely.geometry as sg
import numpy as np
from scipy.interpolate import RegularGridInterpolator

from channest.skeletonize import SkeletonizedPolygon, SkeletonizedLayer


class LayerLengths:
    def __init__(
        self, sl: SkeletonizedLayer, map2d: np.ndarray, mean_map_threshold: float
    ):
        self._lengths = [
            _find_skeleton_length(p, map2d, mean_map_threshold)
            for p in sl.skeletonized_polygons
        ]

    @property
    def lengths(self):
        return self._lengths

    def stat(self, func):
        return func(self._lengths)


def _find_skeleton_length(
    sp: SkeletonizedPolygon, map2d: np.ndarray, mean_map_threshold: float
):
    accumulated_length = 0.0
    for piece in sp.pieces:
        coords = piece.coords
        for i in range(len(coords) - 1):
            p1 = coords[i]
            p2 = coords[i + 1]
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            accumulated_length += np.sqrt(dx**2 + dy**2)

    return accumulated_length
