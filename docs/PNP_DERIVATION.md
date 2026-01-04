# PnP Problem

The Perspective-n-Point or "PnP" problem refers to the common computer-vision task of estimating the 6-DOF pose of a camera given a set of 3D-2D point correspondences.

Inputs:
 - 3D locations of known points within the scene (world-space)
 - 2D locations of the scene points as observed in a camera capture (image-space)
 - camera intrinsics model

Outputs
 - position and orientation of the camera that best explains the given observations


## Geometric Intuition

A minimum of n=4 points are required to uniquely determine a solution. However, with only n=3 points, the solution set is comprised of a small number of discrete poses, from which the correct solution can usually be chosen using some heuristic based on external knowledge. For this reason P3P is often treated as the minimal form (e.g. for use in RANSAC model-fitting).

The following table describes the general solution set for n = 1 through n = 4:

| Number of points ($n \in \mathbb{N}$) | PnP solution set (camera pose $\in SE(3)$)                                                                                                                                                                                                                              |
|---------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 1                                     | **Position**: completely unconstrained<br>**Orientation**: constrained to a 1D subspace of $SO(3)$ for a given position (rotation about the bearing ray)                                                                                                                        |
| 2                                     | **Position**: constrained to a finite 2D surface created by rotating a limaçon-type curve about the axis connecting the 2 points<br>**Orientation**: uniquely determined for a given position                                                                                          |
| 3                                     | **Position**: one of 2 discrete points resulting from the intersection of the 3 pairwise P2P solution sets (both on the same side of the triangle's plane, but not trivially similar to one another)<br>**Orientation**: uniquely determined for a given position |
| 4                                     | **Position**: uniquely determined<br>**Orientation**: uniquely determined                                                                                                                                                                                                       |

Note: The above table assumes that the camera chirality constraint (i.e. the requirement that the observed points appear *in front* of the imager) is enforced. If this is relaxed, the number of possible solutions is generally doubled, which is why much of the literature states that P3P actually has up to *4* possible solutions (the 2 described above, plus their reflections about the optical center).


## Algebraic Solution (P3P)

The algebraic solution to the P3P problem is a classical derivation, typically attributed to Grunert (1841). We present the full derivation here, with the help of the `sympy` computer algebra system for most of the heavy lifting later on.

### Notation

 - Let $X_1, X_2, X_3 \in \mathbb{R}^3$ be the world-space points
 - Let $x_1, x_2, x_3 \in \mathbb{R}^2$ be the image-space observations
 - Let $\pi: \mathbb{R}^3 \to \mathbb{R}^2$ be the camera projection function
 - Let $[R \mid t] \in SE(3)$ be the 6-DOF pose of the camera in world-space

The P3P problem is to find $T = [R \mid t]$ such that:

$$\pi(T^{-1} X_i) = x_i$$

for $i = 1, 2, 3$.


### 1) Tetrahedral Construction

Consider the tetrahedon formed by the camera center (which we denote $P$) and the 3 world-space points.

Denote the lengths of the tetrahedron's base:
 - $s_{12} = \lVert X_1 - X_2 \rVert$
 - $s_{13} = \lVert X_1 - X_3 \rVert$
 - $s_{23} = \lVert X_2 - X_3 \rVert$

Denote the lengths of the tetrahedron's legs (the distances from camera center to world point):
 - $d_1 = \lVert P - X_1 \rVert$
 - $d_2 = \lVert P - X_2 \rVert$
 - $d_3 = \lVert P - X_3 \rVert$

Denote the angles subtended by the tetrahedron's legs:
 - $\theta_{12} = \angle X_1PX_2$
 - $\theta_{13} = \angle X_1PX_3$
 - $\theta_{23} = \angle X_2PX_3$

The $s_{ij}$ can be computed directly from the givens. The $\theta_{ij}$ can also be derived from the givens by taking the bearing-vectors $b_i = \pi^{-1}(x_i)$ and applying the usual angle-between-vectors formula:

$$\cos \theta_{ij} = \pi^{-1}(x_i) \cdot \pi^{-1}(x_j)$$

This leaves only the $d_i$ as the 3 unknowns. Once these are obtained, the full geometry is known and the actual camera pose can be directly computed via rigid-body transform.


### 2) Law-of-Cosines Constraints

Consider the three triangular faces involving $P$. From the law-of-cosines, we have the following constraints on the $d_i$:

$$s_{12}^2 = d_1^2 + d_2^2 - 2d_1d_2\cos \theta_{12}$$
$$s_{13}^2 = d_1^2 + d_3^2 - 2d_1d_3\cos \theta_{13}$$
$$s_{23}^2 = d_2^2 + d_3^2 - 2d_2d_3\cos \theta_{23}$$

This is a system of thee polynomial equations in three unknowns. 


### 3) Eliminating Scale

At this point, we (temporarily) eliminate scale from the problem by rewriting the system above purely in terms of distance ratios. This helps to isolate the geometric structure part of the problem (the difficult part). After that has been solved, scale can be easily recovered.

The literature typically makes the change of variables:
 - $u = d_2/d_1$
 - $v = d_3/d_1$

To obtain the scale-free system, first divide through by $d_1^2$:

$$\begin{align}
\frac{s_{12}^2}{d_1^2} &= 1 + u^2 - 2u\cos \theta_{12} \\
\frac{s_{13}^2}{d_1^2} &= 1 + v^2 - 2v\cos \theta_{13} \\
\frac{s_{23}^2}{d_1^2} &= u^2 + v^2 - 2uv\cos \theta_{23}
\end{align}$$

Then divide (1) by (3), and divide (2) by (3), yielding a system with one fewer degree of freedom:

$$\begin{align*}
\frac{s_{12}^2}{s_{23}^2} &= \frac{1 + u^2 - 2u\cos \theta_{12}}{u^2 + v^2 - 2uv\cos \theta_{23}} \\
\frac{s_{13}^2}{s_{23}^2} &= \frac{1 + v^2 - 2v\cos \theta_{13}}{u^2 + v^2 - 2uv\cos \theta_{23}} \\
\end{align*}$$

We now have a system of two polynomial equations in two unknowns.


### 4) Solving the Reduced System

From here, our reduced system (2 quadratics in 2 unkowns) can be solved numerically through direct optimization.

However, the classic derivation takes the analysis one step further by reducing the system to a single polynomial (1 quartic in 1 unknown). This polynomial can then be solved numerically using the standard eigenvalue-based computation.

Different authors have completed this algebra exercise differently. It is rather length and un-enlightening, so we simply let `sympy.resultant()` do all the heavy lifting:

```python
import sympy as sp

# Known world triangle sides
s12, s13, s23 = sp.symbols("s12 s13 s23", positive=True)

# Known angles between bearing vectors
m12, m13, m23 = sp.symbols("m12 m13 m23", real=True)  # cos(theta_12), cos(theta_12), cos(theta_23)

# Distance ratios u = d2/d1, v = d3/d1
u, v = sp.symbols("u v", real=True, positive=True)

# Reduced system of quadratics
eq1 = s23**2 * (1 + u**2 - 2*u*m12) - s12**2 * (u**2 + v**2 - 2*u*v*m23)
eq2 = s23**2 * (1 + v**2 - 2*v*m13) - s13**2 * (u**2 + v**2 - 2*u*v*m23)

# Reduce to a single polynomial in u
quartic_u = sp.resultant(eq1, eq2, v)

# Print out the expressions for the coefficients
poly = sp.Poly(quartic_u, u)
assert(poly.degree() == 4)

for i, c in enumerate(poly.coeffs()):
    print(f"A{poly.degree()-i} = {poly.coeffs()[i]}")
```

This gives the following coefficients for the quartic polynomial $q(u) = A_4 u^4 + A_3 u^3 + A_2 u^2 + A_1 u + A_0$

```
A4 = -4*m23**2*s12**2*s13**2*s23**4 + s12**4*s23**4 + 2*s12**2*s13**2*s23**4 - 2*s12**2*s23**6 + s13**4*s23**4 - 2*s13**2*s23**6 + s23**8
A3 = 8*m12*m23**2*s12**2*s13**2*s23**4 - 4*m12*s12**2*s13**2*s23**4 + 4*m12*s12**2*s23**6 - 4*m12*s13**4*s23**4 + 8*m12*s13**2*s23**6 - 4*m12*s23**8 - 4*m13*m23*s12**4*s23**4 + 4*m13*m23*s12**2*s13**2*s23**4 + 4*m13*m23*s12**2*s23**6
A2 = 4*m12**2*s13**4*s23**4 - 8*m12**2*s13**2*s23**6 + 4*m12**2*s23**8 - 8*m12*m13*m23*s12**2*s13**2*s23**4 - 8*m12*m13*m23*s12**2*s23**6 + 4*m13**2*s12**4*s23**4 - 4*m13**2*s12**2*s23**6 + 4*m23**2*s12**4*s23**4 - 4*m23**2*s12**2*s13**2*s23**4 - 2*s12**4*s23**4 + 2*s13**4*s23**4 - 4*s13**2*s23**6 + 2*s23**8
A1 = 8*m12*m13**2*s12**2*s23**6 + 4*m12*s12**2*s13**2*s23**4 - 4*m12*s12**2*s23**6 - 4*m12*s13**4*s23**4 + 8*m12*s13**2*s23**6 - 4*m12*s23**8 - 4*m13*m23*s12**4*s23**4 + 4*m13*m23*s12**2*s13**2*s23**4 + 4*m13*m23*s12**2*s23**6
A0 = -4*m13**2*s12**2*s23**6 + s12**4*s23**4 - 2*s12**2*s13**2*s23**4 + 2*s12**2*s23**6 + s13**4*s23**4 - 2*s13**2*s23**6 + s23**8
```

### 5) Recovering the Full Solution

Once we have solved the reduced system (or identified the non-extraneous root(s) of the quartic polynomial), $d_1$, $d_2$, and $d_3$ can be easily recovered from the equations above. With these in hand, the relative geometry of the scene is now fully understood.

To solve for the absolute pose of the camera, we compute the estimated location of the scene points in the camera frame:
 - $\hat{X}_1 = d_1 b_1$
 - $\hat{X}_2 = d_2 b_2$
 - $\hat{X}_3 = d_3 b_3$

where $b_i$ are the bearing vectors computed above. Finally, we estimate the rigid-body similarity transform that maps the $\hat{X}_i$ onto the $X_i$ via the standard Kabsch-Umeyama algorithm.
