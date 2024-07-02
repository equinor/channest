import shapely.geometry as sg
import numpy as np
from scipy.interpolate import RegularGridInterpolator

from channest.skeletonize import SkeletonizedPolygon, SkeletonizedLayer


class LayerLengths:
    def __init__(
        self,
        sl: SkeletonizedLayer,
        map2d: np.ndarray,
        mean_map_threshold: float,
        method: str = "skeleton",
    ):
        if method == "skeleton":
            self._lengths = [
                _find_skeleton_length(p, map2d, mean_map_threshold)
                for p in sl.skeletonized_polygons
            ]
        elif method == "box":
            self._lengths = [
                _find_bounding_box_length(p.polygon) for p in sl.skeletonized_polygons
            ]
        else:
            raise ValueError(f"'method' must be 'skeleton' or 'box', not {method}")

    @property
    def lengths(self):
        return self._lengths

    def stat(self, func):
        return func(self._lengths)


def _find_skeleton_length(
    sp: SkeletonizedPolygon, map2d: np.ndarray, mean_map_threshold: float
):
    backbone = sp.main_channel()
    backbone_coords = np.array(backbone.coords)
    accumulated_length = 0.0

    # Extend start of backbone back to intersection with polygon
    p1 = backbone.coords[0]
    p2 = backbone.coords[1]
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    p0 = p1 - np.array([dx, dy]) * 1000
    intersection_point = sp.polygon.intersection(sg.LineString([p0, p1]))
    if intersection_point.type == "Point":
        backbone_coords = np.insert(
            backbone_coords, 0, intersection_point.coords[0], axis=0
        )
    elif intersection_point.type == "LineString":
        # Take the midpoint of the intersection line
        x0, y0 = intersection_point.coords[0]
        x1, y1 = intersection_point.coords[1]
        xmid, ymid = (x0 + x1) / 2, (y0 + y1) / 2
        backbone_coords = np.insert(backbone_coords, 0, [xmid, ymid], axis=0)

    # Extend end of backbone forward to intersection with polygon
    p1 = backbone.coords[-1]
    p2 = backbone.coords[-2]
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    p0 = p1 - np.array([dx, dy]) * 1000
    intersection_point = sp.polygon.intersection(sg.LineString([p1, p0]))
    if intersection_point.type == "Point":
        backbone_coords = np.append(
            backbone_coords, [intersection_point.coords[0]], axis=0
        )
    elif intersection_point.type == "LineString":
        x0, y0 = intersection_point.coords[0]
        x1, y1 = intersection_point.coords[1]
        xmid, ymid = (x0 + x1) / 2, (y0 + y1) / 2
        backbone_coords = np.append(backbone_coords, [[xmid, ymid]], axis=0)

    for i in range(len(backbone_coords) - 1):
        p1 = backbone_coords[i]
        p2 = backbone_coords[i + 1]
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        accumulated_length += np.sqrt(dx**2 + dy**2)

    return accumulated_length


def _find_bounding_box_length(po: sg.Polygon) -> float:
    mbr_points = list(zip(*po.minimum_rotated_rectangle.exterior.coords.xy))
    mbr_lengths = [
        sg.LineString((mbr_points[i], mbr_points[i + 1])).length
        for i in range(len(mbr_points) - 1)
    ]
    return max(mbr_lengths)
