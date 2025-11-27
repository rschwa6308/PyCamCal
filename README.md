# PyCamCal

General-purpose camera calibration library

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

In the cases where multiple aspects are unknown, they must be jointly solved for using the available data. The so-called "auto-calibration" problem is effectively a joint SLAM + Calibration task (SLAMAC).

## Simulation

This library includes a minimal raycaster (built on top of Open3D) for geometrically-accurate simulated camera captures.

![Cornell box + checkerboard target demo render](media/cornell_box_checkerboard_render.png)

Using this engine we can prepare a scene containg a simulated calibration target and simulate a variety of captures using a provided camera model. The result is a camera calibration test dataset with known ground-truth reference. See [examples/dataset_generation](examples/dataset_generation.ipynb).

![Synthetic calibration dataset renders](media/synthetic_calibration_dataset_image_grid.png)

In addition to the simulated camera images, the dataset includes ground truth for the camera model ([true_camera_model.json](examples/datasets/example_synthetic_calibration_dataset/true_camera_model.json)) and camera poses ([true_camera_poses.csv](examples/datasets/example_synthetic_calibration_dataset/true_camera_poses.csv)).

![Synthetic calibration dataset camera poses](media/synthetic_calibration_dataset_poses.png)
