"""_summary_
"""

import numpy as np
from scipy.spatial.transform import Rotation

from .postprocessing import World2ImgCoordConverter


def world_grasps_ax(
    ax,
    backgound,
    grasp_center,
    grasp_quat,
    cam_intrinsics,
    cam_rot,
    cam_pos,
):
    ax.imshow(backgound)

    world2img_converter = World2ImgCoordConverter()

    center_img = world2img_converter(grasp_center, cam_intrinsics, cam_rot, cam_pos)
    ax.scatter(x=center_img[0], y=center_img[1])

    grasp_rot = Rotation.from_quat(grasp_quat[[1, 2, 3, 0]]).as_matrix()
    grasp_axis = grasp_rot[:, 1].flatten()
    grasp_axis = grasp_axis / np.linalg.norm(grasp_axis)

    antipodal_points_world = np.array(
        [
            grasp_center + grasp_axis * 0.1,
            grasp_center - grasp_axis * 0.1,
        ]
    )
    antipodal_points_img = np.array(
        [
            world2img_converter(p, cam_intrinsics, cam_rot, cam_pos)
            for p in antipodal_points_world
        ]
    )
    ax.plot(antipodal_points_img[:, 0], antipodal_points_img[:, 1])
