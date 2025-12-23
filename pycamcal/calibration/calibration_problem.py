from abc import ABC
from dataclasses import dataclass
from typing import Type
import numpy as np

from ..primitives.pose import Pose3D, R3D
from ..camera_model.camera_model import CameraModel

from ..optimization.optimization_quantity import OptimizationQuantity, Fixed, Unknown


@dataclass
class ImageCaptureObservations:
    "Observations of known scene points from a single camera capture"

    # sub-pixel locations of detected target features in the image, keyed by feature ID
    feature_detections: dict[str, (float, float)]


class CalibrationProblem:
    "General camera calibration problem"

    def __init__(self):
        # camera models to be calibrated, keyed by camera ID
        self.cameras: dict[str, CameraModel] = {}

        # 3D location of feature points within the scene (e.g. calibration target points), keyed by feature ID
        self.scene_points: dict[str, (float, float, float)] = {}

        # camera poses per image to be solved for, keyed by (camera ID, image ID)
        self.camera_poses: dict[(str, str), Pose3D] = {}

        # image capture observations of scene points, keyed by (camera ID, image ID)
        self.observations: dict[(str, str), ImageCaptureObservations] = {}

    def add_camera(self, camera_id: str, camera_model: CameraModel):
        self.cameras[camera_id] = camera_model

    def add_known_scene_points(self, points: dict[str, (float, float, float)]):
        for point_id, point in points.items():
            if point_id in self.scene_points:
                raise ValueError(f"An entry with ID {point_id} already exists within the scene points database")

            self.scene_points[point_id] = Fixed(point)

    def add_observations(self, camera_id: str, image_id: str, observations: ImageCaptureObservations, camera_pose: OptimizationQuantity[Pose3D] | None = None):
        if camera_id not in self.cameras.keys():
            raise ValueError(f"Unkown camera ID {camera_id}")

        self.observations[(camera_id, image_id)] = observations

        if camera_pose is None:
            # convenience default
            camera_pose = Unknown(Pose3D.identity())

        self.camera_poses[(camera_id, image_id)] = camera_pose

    def collect_unknowns(self) -> list[Unknown]:
        unknowns = []

        # consistent ordering
        camera_ids = sorted(self.cameras.keys())
        pose_ids = sorted(self.camera_poses.keys())

        # camera model unknowns
        for camera_id in camera_ids:
            unknowns.extend(self.cameras[camera_id].collect_unknowns())

        # camera pose unknowns
        for pose_id in pose_ids:
            pose = self.camera_poses[pose_id]
            if isinstance(pose, Unknown):
                unknowns.append(pose)

        # scene point unknowns
        # TODO

        return unknowns
    

    def get_residuals(self) -> np.ndarray:
        "Get vector of reprojection residuals. Supports auto-diff."

        def diff(uv_detect, uv_reproj):
            print(f"diff(detect={uv_detect}, reproj={uv_reproj})")
            return np.linalg.norm(uv_detect - uv_reproj, axis=-1)
        
        features_detected, featured_reprojected = self.get_reprojections()

        residuals = diff(features_detected, featured_reprojected)
        return residuals
    

    def get_reprojections(self) -> np.ndarray:
        "Get matching arrays of uv_detected, uv_reprojected"

        features_detected = []
        features_reprojected = []

        # iterate over image captures
        for (cam_id, img_id), obs in self.observations.items():
            cam = self.cameras[cam_id]
            cam_pose = self.camera_poses[(cam_id, img_id)].value()

            # iterate over observed features
            for feat_id, feat_uv_detect in obs.feature_detections.items():
                # transform feature point into camera frame
                feat_pos_world = self.scene_points[feat_id].value()
                feat_pos_camera = cam_pose.inv().apply(feat_pos_world)
                # print(feat_pos_world, "=>", feat_pos_camera)

                # project into image
                feat_uv_reproj = cam.project_into_image(feat_pos_camera)[0]

                # compute reprojection error
                features_detected.append(feat_uv_detect)
                features_reprojected.append(feat_uv_reproj)

        return np.array(features_detected), np.array(features_reprojected)

    def __repr__(self) -> str:
        return f"CalibrationProblem(cameras={self.cameras}, scene_points={self.scene_points}, poses={self.camera_poses}, observations={self.observations})"
