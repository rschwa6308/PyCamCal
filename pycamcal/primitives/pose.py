from dataclasses import dataclass

import jax.numpy as jnp
from jax import tree_util

from .rotation import Rotation3D as R3D

@tree_util.register_pytree_node_class
class Pose3D:
    """
    JAX-friendly representation of an element of SE(3).
    """

    def __init__(self, t: jnp.array, R: R3D):
        self.t = jnp.array(t).reshape((3,))
        self.R = R
    
    # ---------------- PyTree ----------------

    def tree_flatten(self):
        return (self.t, self.R), None

    @classmethod
    def tree_unflatten(cls, aux, children):
        (t, R) = children
        return cls(t, R)
    
    # ---------------- Public API ----------------

    def as_transformation_matrix(self) -> jnp.ndarray:
        T = jnp.vstack([
            jnp.concatenate([self.R.as_matrix(), self.t[:, None]], axis=1),
            jnp.array([[0., 0., 0., 1.]], dtype=jnp.float64)
        ])
        return T

    @staticmethod
    def identity() -> "Pose3D":
        return Pose3D(
            jnp.array([0.0, 0.0, 0.0]),
            R3D.identity()
        )

    def inv(self) -> "Pose3D":
        R_inv = self.R.inv()
        t_inv = -R_inv.apply(self.t)
        return Pose3D(t_inv, R_inv)

    def apply(self, v: jnp.ndarray) -> jnp.ndarray:
        v = jnp.array(v)
        return self.R.apply(v) + self.t
