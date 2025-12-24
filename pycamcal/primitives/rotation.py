from dataclasses import dataclass
import jax.numpy as jnp
from jax import tree_util


def skew(v: jnp.ndarray) -> jnp.ndarray:
    return jnp.array([
        [0.0,   -v[2],  v[1]],
        [v[2],   0.0,  -v[0]],
        [-v[1],  v[0],  0.0],
    ])


def apply_rotvec(r: jnp.ndarray, p: jnp.ndarray) -> jnp.ndarray:
    theta = jnp.linalg.norm(r)

    def small():
        # First-order Taylor
        return p + jnp.cross(r, p)

    def general():
        k = r / theta
        ct = jnp.cos(theta)
        st = jnp.sin(theta)
        return (
            p * ct
            + jnp.cross(k, p) * st
            + k * jnp.dot(k, p) * (1.0 - ct)
        )

    return jnp.where(theta < 1e-8, small(), general())


@tree_util.register_pytree_node_class
class Rotation3D:
    """
    JAX-friendly representation of an element of SO(3).
    Stored as rotation vector internally.
    Designed as a drop-in replacement for `scipy.spatial.transform.Rotation3D`
    """

    # ---------------- PyTree ----------------

    def tree_flatten(self):
        return (self.r,), None

    @classmethod
    def tree_unflatten(cls, aux, children):
        (r,) = children
        return cls(r)

    # ---------------- Constructors ----------------

    def __init__(self, r):
        self.r = jnp.array(r)   # (3,) rotation vector

    @staticmethod
    def identity() -> "Rotation3D":
        return Rotation3D(jnp.zeros(3))

    @staticmethod
    def from_rotvec(r: jnp.ndarray) -> "Rotation3D":
        return Rotation3D(r)

    @staticmethod
    def from_quat(q: jnp.ndarray) -> "Rotation3D":
        """
        q = [x, y, z, w] (same convention as SciPy)
        """
        q = jnp.array(q)

        q = q / jnp.linalg.norm(q)
        v = q[:3]
        w = q[3]
        theta = 2.0 * jnp.arctan2(jnp.linalg.norm(v), w)

        def small():
            return 2.0 * v

        def general():
            return theta * v / jnp.sin(theta / 2.0)

        r = jnp.where(theta < 1e-8, small(), general())
        return Rotation3D(r)

    @staticmethod
    def from_matrix(R: jnp.ndarray) -> "Rotation3D":
        tr = jnp.trace(R)
        w = jnp.sqrt(1.0 + tr) / 2.0
        v = jnp.array([
            R[2, 1] - R[1, 2],
            R[0, 2] - R[2, 0],
            R[1, 0] - R[0, 1],
        ]) / (4.0 * w)

        q = jnp.concatenate([v, jnp.array([w])])
        return Rotation3D.from_quat(q)

    @staticmethod
    def from_euler(seq: str, angles: jnp.ndarray, degrees=False) -> "Rotation3D":
        """
        - `seq`: axis ordering e.g. "xyz", "zyx" (intrinsic, SciPy-compatible)
        - `angles`: (3,) radians
        """

        if degrees:
            angles = jnp.deg2rad(angles)

        def rot_x(a):
            return jnp.array([a, 0.0, 0.0])

        def rot_y(a):
            return jnp.array([0.0, a, 0.0])

        def rot_z(a):
            return jnp.array([0.0, 0.0, a])

        axis_map = {
            "x": rot_x,
            "y": rot_y,
            "z": rot_z,
        }

        r = jnp.zeros(3)
        for ax, ang in zip(seq, angles):
            r = r + axis_map[ax](ang)

        return Rotation3D(r)

    # ---------------- Public API ----------------

    def apply(self, p: jnp.ndarray) -> jnp.ndarray:
        """Apply rotation to a 3D point"""
        return apply_rotvec(self.r, p)

    def inv(self) -> "Rotation3D":
        return Rotation3D(-self.r)

    # def compose(self, other: "Rotation3D") -> "Rotation3D":
    #     """
    #     Composition: self ∘ other
    #     (apply other, then self)

    #     Uses first-order BCH (sufficient for optimization).
    #     """
    #     r1 = self.r
    #     r2 = other.r
    #     return Rotation3D(r1 + r2)

    # ---------------- Conversions ----------------

    def as_rotvec(self) -> jnp.ndarray:
        """Axis-angle (rotation vector)."""
        return self.r

    def as_quat(self) -> jnp.ndarray:
        """
        Quaternion [x, y, z, w] (SciPy convention).
        """
        r = self.r
        theta = jnp.linalg.norm(r)

        def small():
            return jnp.array([0.0, 0.0, 0.0, 1.0])

        def general():
            half = 0.5 * theta
            k = r / theta
            xyz = k * jnp.sin(half)
            w = jnp.cos(half)
            return jnp.concatenate([xyz, jnp.array([w])])

        return jnp.where(theta < 1e-8, small(), general())

    def as_matrix(self) -> jnp.ndarray:
        r = self.r
        theta = jnp.linalg.norm(r)

        def small():
            return jnp.eye(3) + _skew(r)

        def general():
            k = r / theta
            K = _skew(k)
            return (
                jnp.eye(3)
                + jnp.sin(theta) * K
                + (1.0 - jnp.cos(theta)) * (K @ K)
            )

        return jnp.where(theta < 1e-8, small(), general())

    def as_euler(self, order: str = "xyz", degrees=False) -> jnp.ndarray:
        """
        Euler angles (radians).
        Supported orders: 'xyz', 'zyx'
        """

        R = self.as_matrix()

        if order == "xyz":
            sy = jnp.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
            x = jnp.arctan2(R[2, 1], R[2, 2])
            y = jnp.arctan2(-R[2, 0], sy)
            z = jnp.arctan2(R[1, 0], R[0, 0])
            angles = jnp.array([x, y, z])

        elif order == "zyx":
            sy = jnp.sqrt(R[2, 2] ** 2 + R[1, 2] ** 2)
            x = jnp.arctan2(R[1, 2], R[2, 2])
            y = jnp.arctan2(-R[0, 2], sy)
            z = jnp.arctan2(R[0, 1], R[0, 0])
            angles = jnp.array([x, y, z])

        else:
            raise ValueError(f"Unsupported Euler order: {order}")
        
        if degrees:
            angles = jnp.rad2deg(angles)
        
        return angles
