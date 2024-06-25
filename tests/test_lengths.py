from channest.skeletonize import SkeletonizedPolygon
from channest.lengths import _find_skeleton_length
import shapely.geometry


def test_simple_skeleton():
    a = 20
    b = 3
    threshold = 0.0

    polygon = shapely.geometry.Polygon(
        [
            (0, 0),
            (0, b),
            (a, b),
            (a, 0),
            (0, 0),
        ]
    )
    skeleton = SkeletonizedPolygon(polygon)
    estimated_length = _find_skeleton_length(skeleton, None, threshold)

    assert abs(estimated_length - a) <= abs(b)
