import numpy as np
import open3d

from .materials import *

def create_checkerboard_mesh(num_rows, num_cols, square_size, material_a=MAT_BLACK, material_b=MAT_WHITE) -> open3d.geometry.TriangleMesh:
    """
    Create a rectangular mesh with checkerboard coloring, in the (+x, +y) quadrant of the z=0 plane.

    Define both vertex colors and triangle materials for simplicity.
    """

    vertices = []
    triangles = []
    vertex_colors = []
    triangle_materials = []

    for i in range(num_rows):
        for j in range(num_cols):
            # Bottom-left corner of this square
            x0 = j * square_size
            y0 = i * square_size
            x1 = x0 + square_size
            y1 = y0 + square_size
            
            # Four vertices of the square
            v0 = [x0, y0, 0]
            v1 = [x1, y0, 0]
            v2 = [x1, y1, 0]
            v3 = [x0, y1, 0]
            
            idx = len(vertices)
            vertices.extend([v0, v1, v2, v3])
            
            # Two triangles per square
            triangles.append([idx, idx+1, idx+2])
            triangles.append([idx, idx+2, idx+3])
            
            # Assign color based on checker pattern
            square_mat = material_a if (i + j) % 2 == 0 else material_b
            square_color = lookup_material_color(square_mat)
            vertex_colors.extend([square_color]*4)
            triangle_materials.extend([square_mat]*2)

    mesh = open3d.geometry.TriangleMesh()
    mesh.vertices = open3d.utility.Vector3dVector(np.array(vertices))
    mesh.triangles = open3d.utility.Vector3iVector(np.array(triangles))
    mesh.vertex_colors = open3d.utility.Vector3dVector(np.array(vertex_colors))
    mesh.triangle_material_ids = open3d.utility.IntVector(np.array(triangle_materials, dtype=np.int32))
    
    return mesh
