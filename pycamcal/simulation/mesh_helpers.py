import numpy as np
import open3d



def quad_to_tris(quad_verts: tuple[int, int, int, int]) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    a, b, c, d = quad_verts

    return (a, b, d), (c, d, b)



def create_rectangle_mesh(width: float, length: float) -> open3d.geometry.TriangleMesh:
    w = width / 2.0
    l = length / 2.0

    # 4 vertices
    vertices = np.array([
        [-w, -l, 0.0],
        [ w, -l, 0.0],
        [ w,  l, 0.0],
        [-w,  l, 0.0],
    ], dtype=np.float64)

    # 2 triangles
    triangles = np.array([
        [0, 1, 2],
        [0, 2, 3],
    ], dtype=np.int32)

    mesh = open3d.geometry.TriangleMesh(
        vertices=open3d.utility.Vector3dVector(vertices),
        triangles=open3d.utility.Vector3iVector(triangles),
    )

    return mesh


def paint_uniform_material(mesh: open3d.geometry.TriangleMesh, mat_id: int):
    "Set all triangle material ID's to `mat_id`"
    num_tris = len(np.asarray(mesh.triangles))
    mesh.triangle_material_ids = open3d.utility.IntVector(np.full(num_tris, mat_id, dtype=np.int32))
