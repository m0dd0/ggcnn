from nptyping import NDArray, Shape, Float, Int
import numpy as np


class World2ImgCoordConverter:
    def __call__(
        self,
        p_world: NDArray[Shape["3"], Float],
        cam_intrinsics: NDArray[Shape["3,3"], Float],
        cam_rot: NDArray[Shape["3,3"], Float],
        cam_pos: NDArray[Shape["3"], Float],
    ) -> NDArray[Shape["2"], Float]:
        # p_cam = R @ (p_world - T)
        # p_img_h = K @ p_cam = [[p_ix*p_cz]
        #                        [p_iy*p_cz]
        #                        [p_cz     ]]
        # p_img = [[p_ix]  = (p_img_h / p_cz)[:2] = (p_img_h / p_img_h[2])[:2]
        #           p_iy]]

        cam_pos = cam_pos.reshape((3, 1))  # (3,1)

        p_world = p_world.reshape((3, 1))  # (3,1)
        p_cam = cam_rot @ (p_world - cam_pos)
        p_img_h = cam_intrinsics @ p_cam
        p_img = (p_img_h / p_img_h[2])[:2].flatten()  # (2,)

        return p_img


class Img2WorldCoordConverter:
    def __call__(
        self,
        p_img: NDArray[Shape["2"], Int],
        p_cam_z: float,
        cam_intrinsics: NDArray[Shape["3,3"], Float],
        cam_rot: NDArray[Shape["3,3"], Float],
        cam_pos: NDArray[Shape["3"], Float],
    ) -> NDArray[Shape["3"], Float]:
        # K = [[fx 0  cx]
        #      [0  fy cy]
        #      [0  0  1 ]]

        # p_cam = R @ (p_world - T) = [[p_cx] <--> p_world = (inv(R) @ p_cam) + T
        #                             [p_cy]
        #                             [p_cz]]
        # p_img_h = K @ p_cam = [[fx*p_cx + cx*p_cz]  = [[fx*p_cx/p_cz + cx] * p_cz
        #                        [fy*p_cy + cy*p_cz]     [fy*p_cy/p_cz + cy]
        #                        [p_cz             ]]    [1             ]]
        # p_img = [[p_ix]  = [[fx*p_cx/p_cz + cx]   <--> p_cam = [[p_cx]  = [[(p_ix - cx)*p_cz/fx]
        #          [p_iy]]    [fy*p_cy/p_cz + cy]]                [p_cy]     [(p_iy - cy)*p_cz/fy]
        #                                                         [p_cz]]    [p_cz]]
        cam_pos = cam_pos.reshape((3, 1))
        cam_rot_inv = np.linalg.inv(cam_rot)

        p_img = p_img.reshape(2, 1)
        cx = cam_intrinsics[0, 2]
        cy = cam_intrinsics[1, 2]
        fx = cam_intrinsics[0, 0]
        fy = cam_intrinsics[1, 1]
        p_img_x = p_img[0, 0]
        p_img_y = p_img[1, 0]

        p_cam_x = (p_img_x - cx) * p_cam_z / fx
        p_cam_y = (p_img_y - cy) * p_cam_z / fy
        p_cam = np.array([p_cam_x, p_cam_y, p_cam_z]).reshape((3, 1))

        p_world = cam_rot_inv @ p_cam + cam_pos

        p_world = p_world.flatten()

        return p_world
