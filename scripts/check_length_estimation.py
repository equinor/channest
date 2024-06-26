import numpy as np
import plotly.graph_objects as go
import shapely.geometry
import tqdm

from channest.skeletonize import SkeletonizedPolygon
from channest.lengths import _find_skeleton_length


class RandomEllipse:
    def __init__(self, n):
        self.x0 = 100
        self.y0 = 100
        self.a = np.random.uniform(5, 10)
        self.b = np.random.uniform(15, 20)
        self.coords = self.__call__(np.linspace(0, 2 * np.pi, n))

    def coord_pairs(self):
        xs, ys = self.coords
        return [(x, y) for x, y in zip(xs, ys)] + [(xs[0], ys[0])]

    @property
    def length(self):
        return 2 * max(self.a, self.b)

    def __call__(self, t):
        x = self.a * np.cos(t) + np.full_like(t, self.x0)
        y = self.b * np.sin(t) + np.full_like(t, self.y0)
        return x, y


class RandomRectangle:
    def __init__(self):
        self.x0 = 100
        self.y0 = 100
        self.a = np.random.uniform(40, 60)
        self.b = np.random.uniform(5, 10)
        self.coords = self.__call__()

    def coord_pairs(self):
        xs, ys = self.coords
        return [(x, y) for x, y in zip(xs, ys)] + [(xs[0], ys[0])]

    @property
    def length(self):
        return 2 * max(self.a, self.b)

    def __call__(self):
        x = np.array(
            [
                self.x0 - self.a,
                self.x0 + self.a,
                self.x0 + self.a,
                self.x0 - self.a,
                self.x0 - self.a,
            ]
        )
        y = np.array(
            [
                self.y0 - self.b,
                self.y0 - self.b,
                self.y0 + self.b,
                self.y0 + self.b,
                self.y0 - self.b,
            ]
        )
        return x, y


def main():
    n = 40
    n_objects = 1000
    object_type = "rectangle"
    objects = []

    if object_type == "rectangle":
        for _ in tqdm.tqdm(
            range(n_objects), desc="Generating rectangles", unit="rectangle"
        ):
            objects.append(RandomRectangle())
    elif object_type == "ellipse":
        for _ in tqdm.tqdm(
            range(n_objects), desc="Generating ellipses", unit="ellipse"
        ):
            objects.append(RandomEllipse(n))
    else:
        raise ValueError(f"Unknown object type: {object_type}")

    threshold = 0.0

    length_estimates = []

    for object in tqdm.tqdm(objects, desc="Estimating lengths", unit="object"):
        try:
            coord_pairs = object.coord_pairs()
            polygon = shapely.geometry.Polygon(coord_pairs)
            skeleton = SkeletonizedPolygon(polygon)
            length_estimate = _find_skeleton_length(skeleton, None, threshold)
        except:
            length_estimate = np.nan
        finally:
            length_estimates.append(length_estimate)

    if True:
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=[object.length for object in objects],
                y=length_estimates,
                mode="markers",
                marker=dict(size=5, color="blue"),
                name="length estimates",
            )
        )

        # Add axis labels
        fig.update_xaxes(title_text="True length")
        fig.update_yaxes(title_text="Estimated length")

        x_min = min(object.length for object in objects)
        x_max = max(object.length for object in objects)
        y_min = min(length_estimates)
        y_max = max(length_estimates)

        fig.add_trace(
            go.Scatter(
                x=[x_min, x_max],
                y=[x_min, x_max],
                mode="lines",
                line=dict(color="black", width=2),
                name="y = x",
            )
        )

        fig.show()


if __name__ == "__main__":
    main()
