import numpy as np
from typing import List, Dict, Any

from channest.heights import LayerHeights
from channest.widths import LayerWidths, Widths
from channest.polygonize import HulledPolygons
from channest.skeletonize import SkeletonizedLayer
from nrresqml.resqml import ResQml


def create_summary(
    dxy: float,
    polys: List[HulledPolygons],
    poly_skels: List[SkeletonizedLayer],
    poly_widths: List[LayerWidths],
    flat_widths: List[Widths],
    poly_heights: List[LayerHeights],
    flat_heights: np.ndarray,
    archel_name: str = "channel",
) -> Dict[str, Any]:
    return _calculate_results(
        dxy,
        polys,
        poly_skels,
        poly_widths,
        flat_widths,
        poly_heights,
        flat_heights,
        archel_name,
    )


def summary_stats(observations: np.ndarray[float]) -> Dict[str, Any]:
    n = observations.size
    if n < 2:
        raise ValueError("Need at least two observations to calculate summary stats")

    mu = np.mean(observations)
    sigma = np.std(observations)
    return {
        "mean": mu,
        "sd": sigma,
        "count": n,
        "coefficient of variation": sigma / mu,
        "mean estimate standard error": sigma / np.sqrt(n),
        "min": np.min(observations),
        "max": np.max(observations),
        "percentiles": {
            percentage: np.percentile(observations, percentage)
            for percentage in range(10, 100, 10)
        },
    }


def _calculate_results(
    dxy: float,
    polys: List[HulledPolygons],
    poly_skels: List[SkeletonizedLayer],
    poly_widths: List[LayerWidths],
    flat_widths: List[Widths],
    poly_heights: List[LayerHeights],
    flat_heights: np.ndarray,
    archel_name: str,
) -> Dict[str, Any]:
    results = {}
    # Segment widths
    if len(flat_widths) == 0:
        results[archel_name + "-segment-width"] = {
            "mean": np.nan,
            "sd": np.nan,
            "count": 0,
        }
    else:
        per_segment = np.hstack(
            [np.hstack(w.full_widths) for w in flat_widths if w.full_widths.size > 0]
        )
        results[archel_name + "-segment-width"] = summary_stats(per_segment * dxy)

    # Channel widths
    mean_per_channel = [
        np.mean(w.full_widths) for w in flat_widths if w.full_widths.size > 0
    ]
    results[archel_name + "-width"] = summary_stats(np.array(mean_per_channel) * dxy)

    # Segment heights
    if flat_heights.size == 0:
        results[archel_name + "-segment-height"] = {
            "mean": np.nan,
            "sd": np.nan,
            "count": 0,
        }
    else:
        results[archel_name + "-segment-height"] = summary_stats(flat_heights)

    # Channel heights
    height_per_channel = np.hstack([h.mean_heights() for h in poly_heights])
    h_max = LayerHeights.calculate_max_height(poly_heights)
    mode_per_channel = np.array([h.flat_values_max_mode(h_max) for h in poly_heights])

    if height_per_channel.size == 0:
        results[archel_name + "-height"] = {
            "count": 0,
            "mode-mean": np.nan,
            "mode-sd": np.nan,
        }
    else:
        results[archel_name + "-height"] = summary_stats(height_per_channel)
        results[archel_name + "-height"].update(
            {
                "count": height_per_channel.size,
                "mode-mean": np.nanmean(mode_per_channel),
                "mode-sd": np.nanstd(mode_per_channel),
            }
        )

    return results
