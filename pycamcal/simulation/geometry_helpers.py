import numpy as np
import open3d


# def create_box_with_face_removed(size_x, size_y, size_z, remove_face="pos_z"):
#     """
#     Create a box mesh with one face removed.
#     Faces: "+x", "-x", "+y", "-y", "+z", "-".
#     Box is centered at the origin.
#     """

#     # Create full box
#     box = open3d.geometry.TriangleMesh.create_box(size_x, size_y, size_z)

#     # Shift to center at origin
#     box.translate(np.array([-size_x/2, -size_y/2, -size_z/2]))

#     # Mapping from face name to triangle indices in Open3D's box
#     # (each face is two triangles)
#     face_triangles = {
#         "-z": [0, 1],
#         "+z": [2, 3],
#         "-y": [4, 5],
#         "+y": [6, 7],
#         "-x": [8, 9],
#         "+x": [10, 11],
#     }

#     if remove_face not in face_triangles:
#         raise ValueError(f"Unknown face: {remove_face}")

#     # Remove the two triangles of that face
#     all_tris = np.asarray(box.triangles)
#     keep = np.ones(len(all_tris), dtype=bool)
#     keep[face_triangles[remove_face]] = False
#     box.triangles = open3d.utility.Vector3iVector(all_tris[keep])

#     # Cull UVs
#     if box.triangle_uvs:
#         all_uvs = np.asarray(box.triangle_uvs)
#         # 3 UVs per triangle
#         mask_uv = np.repeat(keep, 3)
#         box.triangle_uvs = open3d.utility.Vector2dVector(all_uvs[mask_uv])

#     box.compute_vertex_normals()

#     return box



def create_planar_rectangle(width_x: float, width_y: float) -> open3d.geometry.TriangleMesh:
    "Create a flat rectangle mesh (consisting of two triangles) in the xy-plane centered at the origin"

    hx = float(width_x) / 2.0
    hy = float(width_y) / 2.0

    # CCW winding when viewed from +Z
    vertices = [
        [-hx, -hy, 0.0],  # 0
        [ hx, -hy, 0.0],  # 1
        [ hx,  hy, 0.0],  # 2
        [-hx,  hy, 0.0],  # 3
    ]
    triangles = [
        [0, 1, 2],  # lower-right triangle
        [0, 2, 3],  # upper-left triangle
    ]

    mesh = open3d.geometry.TriangleMesh()
    mesh.vertices = open3d.utility.Vector3dVector(vertices)
    mesh.triangles = open3d.utility.Vector3iVector(triangles)

    # simple UVs mapping rectangle to [0,1]x[0,1] (3 UVs per triangle)
    mesh.triangle_uvs = open3d.utility.Vector2dVector([
        [0.0, 0.0], [1.0, 0.0], [1.0, 1.0],  # tri 0
        [0.0, 0.0], [1.0, 1.0], [0.0, 1.0],  # tri 1
    ])

    mesh.compute_vertex_normals()
    return mesh


def quad_to_tris(quad_verts: tuple[int, int, int, int]) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    a, b, c, d = quad_verts

    return (a, b, d), (c, d, b)
