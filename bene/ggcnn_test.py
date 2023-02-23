

# import Dataset classes for Cornell and Jaquard 
# (inherit from GraspDatasetbase which inherits from torch.utils.data.Dataset)
from utils.data.cornell_data import CornellDataset
from utils.data.jacquard_data import JacquardDataset

import torch

# import model
from models.ggcnn import GGCNN

# import post processing function (convert raw output GG-CNN to numpy arrays and apply filtering)
from models.common import post_process_output

import logging

# define arguments
weights_path = '/home/i53/student/b_woerz/ggcnn2/ggcnn_weights_cornell/ggcnn_epoch_23_cornell_statedict.pt'
model_path = '/home/i53/student/b_woerz/ggcnn2/ggcnn_weights_cornell/ggcnn_epoch_23_cornell'
dataset_name = 'jaquard'
dataset_path = '/home/i53/student/b_woerz/ggcnn2/Jaquard_Samples'
use_depth = 1       # 'Use Depth image for evaluation (1/0)'
use_rgb = 0         # 'Use RGB image for evaluation (0/1)'
augment = True      # 'Whether data augmentation should be applied')
split = 0.9         # 'Fraction of data for training (remainder is validation)'
ds_rotate = 0.0     # 'Shift the start point of the dataset to use a different test/train split')
num_workers = 8     # 'Dataset workers'
n_grasps = 1        # 'Number of grasps to consider per image')

Dataset = JacquardDataset



model = GGCNN()
model.load_state_dict(torch.load(weights_path,map_location=torch.device('cuda')))
print(model)


if __name__ == '__main__':
    
    # Load Network
    net = torch.load(model_path)
    device = torch.device("cuda:0")

    # Load Dataset
    print('Loading {} Dataset...'.format(dataset_name))
    logging.info('Loading {} Dataset...'.format(dataset_name))
    
    test_dataset = Dataset(dataset_path, start=split, end=1.0, ds_rotate=ds_rotate,
                           random_rotate=augment, random_zoom=augment,
                           include_depth=use_depth, include_rgb=use_rgb)
    
    test_data = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers
    )
    logging.info('Done')

    obj_counter = 0
    with torch.no_grad():
        for idx, (x, y, didx, rot, zoom) in enumerate(test_data):
            logging.info('Processing {}/{}'.format(idx+1, len(test_data)))
            xc = x.to(device)
            yc = [yi.to(device) for yi in y]
            lossd = net.compute_loss(xc, yc)

            q_img, ang_img, width_img = post_process_output(lossd['pred']['pos'], lossd['pred']['cos'],
                                                        lossd['pred']['sin'], lossd['pred']['width'])
            obj_counter += 1
            

print('Position: {}, Type: {}, Shape: {}'.format(q_img, type(q_img), q_img.shape))
print('Angle: Type: {}, Shape: {}'.format(type(ang_img), ang_img.shape))
print('Width: Type: {}, Shape: {}'.format(type(width_img), width_img.shape))
print('Number of objects: {}'.format(obj_counter))