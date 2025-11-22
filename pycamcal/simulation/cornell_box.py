import numpy as np
import open3d

from .geometry_helpers import quad_to_tris, create_planar_rectangle

def create_cornell_box(width=4, depth=4, height=5) -> dict[str, open3d.geometry.TriangleMesh]:
    vertices = np.array([
        (-width/2, -depth/2, 0.0   ),   # 0
        (+width/2, -depth/2, 0.0   ),   # 1
        (+width/2, +depth/2, 0.0   ),   # 2
        (-width/2, +depth/2, 0.0   ),   # 3
        (-width/2, -depth/2, height),   # 4
        (+width/2, -depth/2, height),   # 5
        (+width/2, +depth/2, height),   # 6
        (-width/2, +depth/2, height),   # 7
    ])

    floor_verts      = [0, 1, 2, 3]
    ceiling_verts    = [4, 5, 6, 7][::-1]
    wall_left_verts  = [0, 3, 7, 4]
    wall_right_verts = [1, 2, 6, 5][::-1]
    wall_back_verts  = [3, 2, 6, 7]

    def make_mesh(quad_verts):
        mesh = open3d.geometry.TriangleMesh()
        mesh.vertices = open3d.utility.Vector3dVector(vertices)
        mesh.triangles = open3d.utility.Vector3iVector(quad_to_tris(quad_verts))
        mesh.compute_vertex_normals()
        return mesh
    
    floor      = make_mesh(floor_verts)
    ceiling    = make_mesh(ceiling_verts)
    wall_left  = make_mesh(wall_left_verts)
    wall_right = make_mesh(wall_right_verts)
    wall_back  = make_mesh(wall_back_verts)

    white = [0.7, 0.7, 0.7]
    red   = [1, 0, 0]
    green = [0, 1, 0]
    blue  = [0, 0, 1]

    floor     .paint_uniform_color(white)
    ceiling   .paint_uniform_color(white)
    wall_left .paint_uniform_color(red)
    wall_right.paint_uniform_color(green)
    wall_back .paint_uniform_color(blue)#white)

    geoms = {
        "floor":      floor,
        "ceiling":    ceiling,
        "wall_left":  wall_left,
        "wall_right": wall_right,
        "wall_back":  wall_back
    }
    return geoms

