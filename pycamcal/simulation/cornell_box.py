import numpy as np
import open3d

from .mesh_helpers import paint_uniform_material, quad_to_tris
from .materials import *

def create_cornell_box(width=4, depth=4, height=5, use_vertex_colors=True, use_triangle_material_ids=True) -> dict[str, open3d.geometry.TriangleMesh]:
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

    if use_vertex_colors:
        floor     .paint_uniform_color(lookup_material_color(MAT_GRAY_LIGHT))
        ceiling   .paint_uniform_color(lookup_material_color(MAT_GRAY_LIGHT))
        wall_left .paint_uniform_color(lookup_material_color(MAT_RED  ))
        wall_right.paint_uniform_color(lookup_material_color(MAT_GREEN))
        wall_back .paint_uniform_color(lookup_material_color(MAT_BLUE ))

    if use_triangle_material_ids:
        paint_uniform_material(floor,      MAT_GRAY_LIGHT)
        paint_uniform_material(ceiling,    MAT_GRAY_LIGHT)
        paint_uniform_material(wall_left,  MAT_RED  )
        paint_uniform_material(wall_right, MAT_GREEN)
        paint_uniform_material(wall_back,  MAT_BLUE )

    geoms = {
        "floor":      floor,
        "ceiling":    ceiling,
        "wall_left":  wall_left,
        "wall_right": wall_right,
        "wall_back":  wall_back
    }
    return geoms

