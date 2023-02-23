# import Dataset classes for Cornell and Jaquard 
# (inherit from GraspDatasetbase which inherits from torch.utils.data.Dataset)
from utils.data.cornell_data import CornellDataset
from utils.data.jacquard_data import JacquardDataset
from utils.data.ycb_data import YCBDataset
from utils.dataset_processing.grasp import Grasp

import torch

# import model
from models.ggcnn import GGCNN

# import post processing function (convert raw output GG-CNN to numpy arrays and apply filtering)
from models.common import post_process_output

import logging

import numpy as np
import matplotlib.pyplot as plt

from utils.dataset_processing import grasp, image

# define arguments
weights_path = '/home/i53/student/b_woerz/ggcnn2/ggcnn_weights_cornell/ggcnn_epoch_23_cornell_statedict.pt'
model_path = '/home/i53/student/b_woerz/ggcnn2/ggcnn_weights_cornell/ggcnn_epoch_23_cornell'

depth_img_path = '/home/i53/student/b_woerz/Documents/ycb_sim_data_1/ycb_simulation_sample_002_cracker_box/depth_img.npy'
#'/home/i53/student/b_woerz/Documents/ycb_sim_data_1/ycb_simulation_sample_002_cracker_box/depth_img.npy'
#'/home/i53/student/b_woerz/Documents/ycb_sim_data_1/ycb_simulation_sample_001_master_chef_can/depth_img.npy'

# rotation always 0
rotation = 0
# zoom has to be 1.0 I think - otherwise error
zoom = 1.0 
output_size = 300

# wird dann in simulation durch cam.intrinsics, ersetzt
# [[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1]]
camera_intrinsics = [[560.84610068, 0., 320.], [0., 579.41125497, 240.], [0., 0., 1.]]
# wird dann in simulation durch cam.get_cart_pos_quat() ersetzt
cam_pos = [0.5, 0.0, 1.0]
cam_quat = [0.7071067811865476, 0.0, 0.0, -0.7071067811865475]

model = GGCNN()
model.load_state_dict(torch.load(weights_path,map_location=torch.device('cuda')))


def _get_crop_attrs():
        # TODO: improve function to get center - not hardcoded
        center = (240,320)
        left = max(0, min(center[1] - output_size // 2, 640 - output_size))
        top = max(0, min(center[0] - output_size // 2, 480 - output_size))
        return center, left, top

def preprocessing(depth_img, rot, zoom):
        og_depth_img = image.Image(depth_img.img)

        center, left, top = _get_crop_attrs()
        depth_img.rotate(rot, center)
        depth_img.crop((top, left), (min(480, top + output_size), min(640, left + output_size)))
        depth_img.normalise()
        depth_img.zoom(zoom)
        depth_img.resize((output_size, output_size))

        og_depth_img.rotate(rot, center)
        og_depth_img.crop((top, left), (min(480, top + output_size), min(640, left + output_size)))
        og_depth_img.zoom(zoom)
        og_depth_img.resize((output_size, output_size))


        return og_depth_img.img, depth_img.img

def numpy_to_torch(s):
        if len(s.shape) == 2:
            return torch.from_numpy(np.expand_dims(s, 0).astype(np.float32))
        else:
            return torch.from_numpy(s.astype(np.float32))

def get_ggcnn_output(weights_path, model_path, depth_img):
    #Load Network
    net = torch.load(model_path)
    device = torch.device("cuda:0")

    with torch.no_grad():
        
        depth_imagec = depth_img.to(device)
            
        pred = net.forward(depth_imagec) #-> pos_output, cos_output, sin_output, width_output
        print('PREDICTION: Type: {}, Shape: {}'.format(type(pred), len(pred)))
        q_img, ang_img, width_img = post_process_output(pred['pos'], pred['cos'], pred['sin'], pred['width'] )
        return q_img, ang_img, width_img
    
def quaternion_to_rotation_matrix(q):
    """
    Convert a quaternion to a rotation matrix.
    Args:
    - q (4-element list or numpy array): A unit quaternion in the form [w, x, y, z].
    Returns:
    - R (3x3 numpy array): A rotation matrix corresponding to the input quaternion.
    """
    w, x, y, z = q
    R = np.array([[1-2*y*y-2*z*z, 2*x*y-2*w*z, 2*x*z+2*w*y],
                  [2*x*y+2*w*z, 1-2*x*x-2*z*z, 2*y*z-2*w*x],
                  [2*x*z-2*w*y, 2*y*z+2*w*x, 1-2*x*x-2*y*y]])
    return R

def euler_to_quaternion(roll, pitch, yaw):
    """
    Convert Euler angles in roll, pitch, yaw format to a quaternion.
    Args:
    - roll (float): Rotation angle around the x-axis, in radians.
    - pitch (float): Rotation angle around the y-axis, in radians.
    - yaw (float): Rotation angle around the z-axis, in radians.
    Returns:
    - q (4-element numpy array): A unit quaternion representing the input orientation in the form [w, x, y, z].
    """
    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)

    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy

    q = np.array([qw, qx, qy, qz])
    q /= np.linalg.norm(q)  # Ensure unit quaternion
    return q

def get_2D_grasp(pose_img, ang_img):
      max_value = np.amax(pose_img)
      max_index = np.argmax(pose_img)
      grasp_point = np.unravel_index(max_index, pose_img.shape)

      grasp_angle = ang_img[grasp_point]

      g = Grasp(grasp_point, grasp_angle)
      return g, max_value

def transform_2D_grasp_to_6D(g, cam_intrinsics, cam_position, cam_quat, depth_img):
    # [[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1]]
    cx = cam_intrinsics[0][2]
    cy = cam_intrinsics[1][2]
    fx = cam_intrinsics[0][0]
    fy = cam_intrinsics[1][1]

    #grasp position in image coordinates
    p_img_x = g.center[0]
    p_img_y = g.center[1]
    
    #grasp position in cam coordinates
    p_cam_z = depth_img[g.center[0]][g.center[1]]
    p_cam_x = (p_img_x - cx) * p_cam_z / fx
    p_cam_y = (p_img_y - cy) * p_cam_z / fy

    p_cam = [p_cam_x, p_cam_y, p_cam_z]
    p_cam = np.array(p_cam).reshape((3, 1))

    #cam_quat in rotationsmatrix umstellen (cam_rot)
    cam_rot = quaternion_to_rotation_matrix(cam_quat)
    #cam_rot invertieren (cam_rot_inv)
    cam_rot_inv = np.linalg.inv(cam_rot)

    #grasp position in world coordinates
    cam_pos = np.array(cam_position).reshape((3, 1))
    p_world = cam_rot_inv @ p_cam + cam_pos
    
    p_world = p_world.flatten()

    #grasp orientation in world coordinates
    q_world = euler_to_quaternion(0, 0, g.angle)

    return p_world, q_world


      
      
      

if __name__ == '__main__':
    
    # loading image -> will not be necessary in simulation
    depth_image = image.DepthImage.from_npy(depth_img_path)
    print('DEPTH IMAGE 1: Type: {}, Shape: {}'.format(type(depth_image), depth_image.shape))
    imgplot1 = plt.imshow(depth_image)
    plt.show()

    #preprocessing image
    og_depth_image, preprocessed_depth_image = preprocessing(depth_image, rotation, zoom)
    print('DEPTH IMAGE 2: Type: {}, Shape: {}'.format(type(preprocessed_depth_image), preprocessed_depth_image.shape))
    imgplot2 = plt.imshow(preprocessed_depth_image)
    plt.show()

    #passing image through ggcnn -> (300 X 300) images for position, angle, width
    depth_image = numpy_to_torch(preprocessed_depth_image)
    q_img, ang_img, width_img = get_ggcnn_output(weights_path, model_path, depth_image)
    print('Position: Type: {}, Shape: {}'.format(type(q_img), q_img.shape))
    print('Angle: Type: {}, Shape: {}'.format(type(ang_img), ang_img.shape))
    print('Width: Type: {}, Shape: {}'.format(type(width_img), width_img.shape))

    #derive performable grasp pose -> position in cartesian coordinates [x,y,z], orientation as quaternion [x0, x1, x2, x3]
    grasp_2D, grasp_quality = get_2D_grasp(q_img, ang_img)
    print('GRASP2D: grasp point: {}, quality: {}, angle: {}'.format(grasp_2D.center, grasp_quality, grasp_2D.angle))
    grasp_position, grasp_orientation = transform_2D_grasp_to_6D(grasp_2D, camera_intrinsics, cam_pos, cam_quat, og_depth_image)
    print('GRASP6D: position: {}{}, orientation: {}{}'.format(type(grasp_position), grasp_position, type(grasp_orientation), grasp_orientation))


    

