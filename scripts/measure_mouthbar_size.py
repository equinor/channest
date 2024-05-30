import numpy as np
import plotly.graph_objects as go
import pathlib
import sys
from skimage import measure
from nrresqml.resqml import ResQml
from channest import polygonize

model_file = sys.argv[1]
model_file_path = pathlib.Path(model_file)
rq = ResQml.read_zipped(model_file_path)

foreground_archel = {"name": "mouth bar", "value": 4}
raw_cube, pillars, grid_params = polygonize.create_cube(rq, None, foreground_archel)

layers_to_merge = 21
cube = polygonize.smoothen_cube(raw_cube, layers_to_merge)

verts, faces, normals, values = measure.marching_cubes(
    cube.transpose((1, 2, 0)), level=0.4
)

ic = np.round(verts[:, 0]).astype(int)
jc = np.round(verts[:, 1]).astype(int)
kc = np.round(verts[:, 2]).astype(int)

xc = ic * grid_params.dx
yc = jc * grid_params.dy
zc = pillars[kc, ic, jc]

fig = go.Figure(
    data=[
        go.Mesh3d(
            x=xc,
            y=yc,
            z=zc,
            i=faces[:, 0],
            j=faces[:, 1],
            k=faces[:, 2],
            intensity=values,
            colorscale="Viridis",
            showscale=True,
        )
    ]
)

fig.update_layout(scene_aspectmode="manual", scene_aspectratio=dict(x=10, y=10, z=1))

fig.show()
