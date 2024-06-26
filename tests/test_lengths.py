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


def test_t_shape_skeleton():
    a = 100
    b = 3
    c = 50
    d = 80
    threshold = 0.0

    polygon = shapely.geometry.Polygon(
        [
            (0, 0),
            (0, b),
            (a, b),
            (a, b + c),
            (a + b, b + c),
            (a + b, b),
            (a + b + d, b),
            (a + b + d, 0),
            (0, 0),
        ]
    )
    skeleton = SkeletonizedPolygon(polygon)
    estimated_length = _find_skeleton_length(skeleton, None, threshold)

    expected_length = a + d

    assert abs(estimated_length - expected_length) <= abs(b)


def test_zero_crossing_rectangle():
    x0 = 0
    a = 100
    b = 3
    threshold = 0.0

    polygon = shapely.geometry.Polygon(
        [
            (x0 - a, 0),
            (x0 + a, 0),
            (x0 + a, b),
            (x0 - a, b),
            (x0 - a, 0),
        ]
    )
    skeleton = SkeletonizedPolygon(polygon)
    estimated_length = _find_skeleton_length(skeleton, None, threshold)

    expected_length = 2 * a

    assert abs(estimated_length - expected_length) <= abs(b)
