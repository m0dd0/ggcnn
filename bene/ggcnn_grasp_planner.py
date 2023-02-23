# import Dataset classes for Cornell and Jaquard 
# (inherit from GraspDatasetbase which inherits from torch.utils.data.Dataset)
from ..utils.data.cornell_data import CornellDataset
from utils.data.jacquard_data import JacquardDataset
from utils.data.ycb_data import YCBDataset
from utils.dataset_processing.grasp import Grasp, Grasp6D
from utils.dataset_processing.grasp import detect_grasps

import torch

# import model
from models.ggcnn import GGCNN

# import post processing function (convert raw output GG-CNN to numpy arrays and apply filtering)
from models.common import post_process_output

import logging

import numpy as np
import matplotlib.pyplot as plt

from utils.dataset_processing import grasp, image

# weights and model -> weights probably not needed
# TODO: define model path universal ./
weights_path = '/home/i53/student/b_woerz/ggcnn2/ggcnn_weights_cornell/ggcnn_epoch_23_cornell_statedict.pt'
model_path = './ggcnn_weights_cornell/ggcnn_epoch_23_cornell'
#'/home/i53/student/b_woerz/ggcnn2/ggcnn_weights_cornell/ggcnn_epoch_23_cornell'

depth_img_path = '/home/i53/student/b_woerz/Documents/ycb_sim_data_1/ycb_simulation_sample_002_cracker_box/depth_img.npy'
#'/home/i53/student/b_woerz/Documents/ycb_sim_data_1/ycb_simulation_sample_002_cracker_box/depth_img.npy'
#'/home/i53/student/b_woerz/Documents/ycb_sim_data_1/ycb_simulation_sample_001_master_chef_can/depth_img.npy'

# rotation always 0
rotation = 0
# zoom = 1 means no zoom
zoom = 1.0 
output_size = 300

# wird dann in simulation durch cam.intrinsics, ersetzt
# [[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1]]
camera_intrinsics_input = [[560.84610068, 0., 320.], [0., 579.41125497, 240.], [0., 0., 1.]]
# wird dann in simulation durch cam.get_cart_pos_quat() ersetzt
cam_pos_input = [0.5, 0.0, 1.0]
cam_quat_input = [0.7071067811865476, 0.0, 0.0, -0.7071067811865475]

 # loading image -> will not be necessary in simulation
depth_image_instance = image.DepthImage.from_npy(depth_img_path)
depth_image = depth_image_instance.img

#determine how many grasps to output
number_grasps = 10

def _get_crop_attrs():
        # TODO: improve function to get center - not hardcoded
        center = (240,320)
        left = max(0, min(center[1] - output_size // 2, 640 - output_size))
        top = max(0, min(center[0] - output_size // 2, 480 - output_size))
        return center, left, top

def preprocessing(depth_img, rot, zoom):
        #TODO: apply mask to get rid of table edge
        og_depth_img = image.DepthImage(depth_img.img)
        
        
        imgplot3 = plt.imshow(depth_img.img)
        plt.show()
        center, left, top = _get_crop_attrs()
        depth_img.rotate(rot, center)
        depth_img.crop((top, left), (min(480, top + output_size), min(640, left + output_size)))
        og_depth_img.crop((top, left), (min(480, top + output_size), min(640, left + output_size)))
        imgplot4 = plt.imshow(depth_img.img)
        plt.show()
        depth_img.mask(mask_value = 1.5, set_value = 1.02)
        imgplot8 = plt.imshow(depth_img.img)
        plt.show()
        depth_img.normalise()
        imgplot5 = plt.imshow(depth_img.img)
        plt.show()

        depth_img.zoom(zoom)
        depth_img.resize((output_size, output_size))
        imgplot5 = plt.imshow(og_depth_img.img)
        plt.show()
        return og_depth_img.img, depth_img.img

def numpy_to_torch(s):
        if len(s.shape) == 2:
            return torch.from_numpy(np.expand_dims(s, 0).astype(np.float32))
        else:
            return torch.from_numpy(s.astype(np.float32))

def get_ggcnn_output(depth_img, model_path = model_path):
    #Load Network
    net = torch.load(model_path)
    device = torch.device("cuda:0")

    with torch.no_grad():
        
        depth_imagec = depth_img.to(device)
            
        pred = net.forward(depth_imagec) #-> pos_output, cos_output, sin_output, width_output
        print('PREDICTION: Type: {}, Shape: {}'.format(type(pred), len(pred)))
        q_img, ang_img, width_img = post_process_output(pred['pos'], pred['cos'], pred['sin'], pred['width'] )
        return q_img, ang_img, width_img

def uncrop_2D_grasps(grasps2D_list):
     #TODO: postrpocess the output images (uncrop)
    crop_center, left, top = _get_crop_attrs()
    for g in grasps2D_list:
        g.center = list(g.center)  # Convert tuple to list
        g.center[0] += top  # update y coordinate
        g.center[1] += left # update x coordinate
        g.center = tuple(g.center)  # Convert list back to tuple
        
    
    return grasps2D_list

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


def transform_2D_grasp_to_6D(g, cam_intrinsics, cam_position, cam_quat):
    # [[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1]]
    cx = cam_intrinsics[0][2]
    cy = cam_intrinsics[1][2]
    fx = cam_intrinsics[0][0]
    fy = cam_intrinsics[1][1]

    #grasp position in image coordinates
    #vielleicht andersrum - erster Wert ist y zweiter ist x
    p_img_x = g.center[1]
    p_img_y = g.center[0]
    
    #grasp position in cam coordinates
    p_cam_z = g.depth
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

def get_6D_grasps(grasps2D_list, camera_intrinsics, cam_pos, cam_quat):
    grasps6D = []
    for g in grasps2D_list:
        p_world, q_world = transform_2D_grasp_to_6D(g, camera_intrinsics, cam_pos, cam_quat)
        g6D = Grasp6D(p_world, q_world)
        grasps6D.append(g6D)
    return grasps6D


def ggcnn_get_grasp(depth_image, camera_intrinsics, cam_pos, cam_quat, rotation=0.0, zoom=1.0, output_size = 300, weights_path = './ggcnn_weights_cornell/ggcnn_epoch_23_cornell_statedict.pt'):

    depth_img_inst = image.DepthImage(depth_image)

    #preprocessing image
    og_depth_img, preprocessed_depth_image = preprocessing(depth_img_inst, rotation, zoom)
    
    print('DEPTH IMAGE 2: Type: {}, Shape: {}'.format(type(preprocessed_depth_image), preprocessed_depth_image.shape))
    imgplot2 = plt.imshow(preprocessed_depth_image)
    #plt.show()

    #passing image through ggcnn -> (300 X 300) images for position, angle, width
    depth_image_tens = numpy_to_torch(preprocessed_depth_image)
    q_img, ang_img, width_img = get_ggcnn_output(depth_image_tens)

    print('Position IMAGE: Type: {}, Shape: {}'.format(type(q_img), q_img.shape))
    imgplot3 = plt.imshow(q_img)
    plt.show()

    #derive performable grasp pose -> position in cartesian coordinates [x,y,z], orientation as quaternion [x0, x1, x2, x3]
    
    # hier wird auch depth Wert initialisiert
    grasps2D = detect_grasps(og_depth_img, q_img, ang_img, no_grasps=number_grasps)
    print('2D GRASP Center: {}, Angle: {}'.format(grasps2D[0].center, grasps2D[0].angle))
    grasps2D = uncrop_2D_grasps(grasps2D)
    print('UNCROPPED 2D GRASP Center: {}'.format(grasps2D[0].center))
    #TODO: Achtung Tiefenwert einfach aus q_img
    grasps6D = get_6D_grasps(grasps2D, camera_intrinsics, cam_pos, cam_quat)
    print('6D GRASP: position: {}, orientation: {}'.format(grasps6D[0].position, grasps6D[0].orientation))
    
    return grasps6D
     

      
      

if __name__ == '__main__':
    
    grasps6D_list = ggcnn_get_grasp(depth_image, camera_intrinsics_input, cam_pos_input, cam_quat_input)
    

    

