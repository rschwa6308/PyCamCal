# PyCamCal

General-purpose camera calibration library [WORK-IN-PROGRESS]

## Camera Models

Supported distortion model:
 - Radial-Tangential (aka Brown-Conrady)
 - Full OpenCV distortion parameters 🚧
 - Generalized warp-table 🚧

## Calibration Types

There are three main types of camera calibration depending on which aspects of the procedure are controlled:
 
|          Type          | Description                                                                                  | Camera Model | Camera Poses | Scene Geometry |
|------------------------|----------------------------------------------------------------------------------------------|--------------|--------------|----------------|
| **Surveyed Calibration**   | Laboratory calibration with a structured calibration target and external metrology for measuring camera poses. | unknown  | known ✓    | known ✓      |
| **Unsurveyed Calibration** | Laboratory/field calibration with a structured calibration target but no metrology.                     | unknown  | unknown  | known ✓      |
| **Auto-Calibration**       | Field calibration of unstructured scene with no metrology.              | unknown  | unknown  | unknown    |

In the cases where multiple aspects are unknown, they must be jointly solved for using the available data. For example, the so-called "auto-calibration" problem is effectively a joint SLAM + Calibration task (SLAMAC).

## Simulation

This library includes a minimal raycaster (built on top of Open3D) for geometrically-accurate simulated camera captures.

![Cornell box + checkerboard target demo render](media/cornell_box_checkerboard_render.png)

Using this engine we can prepare a scene containg a simulated calibration target and simulate a variety of captures using a provided camera model. The result is a camera calibration test dataset with known ground-truth reference. See [examples/dataset_generation](examples/dataset_generation.ipynb).

![Synthetic calibration dataset renders](media/synthetic_calibration_dataset_image_grid.png)

In addition to the simulated camera images, the dataset includes ground truth for the camera model ([true_camera_model.json](examples/datasets/example_synthetic_calibration_dataset/true_camera_model.json)) and camera poses ([true_camera_poses.csv](examples/datasets/example_synthetic_calibration_dataset/true_camera_poses.csv)).

![Synthetic calibration dataset camera poses](media/synthetic_calibration_dataset_poses.png)

## Algorithm Overview

At its core, the calibration algorithm is an optimization problem where the free variables are the camera intrinsics/extrinsics, and the objective is to minimize the total reprojection error from the given point correspondances. For unsurveyed calibration, the algorithm performs the following steps:

1) Compute an initial guess for the intrinsics parameters (based on heuristics and/or user input)
2) Initialize distortion parameters to zero
3) Compute an initial guess for camera poses by solving the [PnP](https://en.wikipedia.org/wiki/Perspective-n-Point) problem for each image (given the above pinhole model)
4) Run a non-linear least squares solver (Levenberg-Marquardt) to refine both camera parameters and poses in order to minimize the total reprojection errors

Symbolically:
 - Let $\pi_{K, \theta}: \mathbb{R}^3 \to \mathbb{R}^2$ be the camera projection function defined by intrinsics parameters $K$ and distortion parameters $\theta$
 - Let $[R_i \mid t_i] \in \mathrm{SE}(3)$ be the 6-DOF pose of the camera in image $i$
 - Let $x_{ij} \in \mathbb{R}^2$ be the observed sub-pixel location of the $j$-th calibration target point in image $i$
 - Let $X_j \in \mathbb{R}^3$ be the 3D location of the $j$-th calibration target point

The objective is thus

$$\min_{K, \theta, R_i, t_i} \sum_{i,j} \mathrm{DIFF}( x_{ij}, x_{ij}' )^2$$

where $x_{ij}' = \pi_{K, \theta}([R_i \mid t_i]^{-1} X_j)$ is the hypothetical location of the calibration point reprojected into the image, and $\mathrm{DIFF}: (\mathbb{R}^2, \mathbb{R}^2) \to \mathbb{R}^+$ is a measure of error between the reprojected point and the observed point. Typically, $\mathrm{DIFF}(x, x') = \lVert x - x' \rVert$ is used.
