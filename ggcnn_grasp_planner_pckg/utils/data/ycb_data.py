import os
import glob

from .grasp_data import GraspDatasetBase
#from utils.dataset_processing import grasp, image
from ..dataset_processing import grasp, image


class YCBDataset(GraspDatasetBase):
    """
    Dataset wrapper for the YCB dataset.
    """
    def __init__(self, file_path, start=0.0, end=1.0, ds_rotate=0, **kwargs):
        """
        :param file_path: YCB Dataset directory.
        :param start: If splitting the dataset, start at this fraction [0,1]
        :param end: If splitting the dataset, finish at this fraction
        :param ds_rotate: If splitting the dataset, rotate the list of items by this fraction first
        :param kwargs: kwargs for GraspDatasetBase
        """
        super(YCBDataset, self).__init__(**kwargs)

        graspf = glob.glob(os.path.join(file_path, '*', '*cam_intrinsics.npy'))
        graspf.sort()
        l = len(graspf)

        if l == 0:
            raise FileNotFoundError('No dataset files found. Check path: {}'.format(file_path))

        if ds_rotate:
            graspf = graspf[int(l*ds_rotate):] + graspf[:int(l*ds_rotate)]

        depthf = [f.replace('cam_intrinsics.npy', 'depth_img.npy') for f in graspf]
        rgbf = [f.replace('depth_img.npy', 'rgb_img.npy') for f in depthf]

        self.grasp_files = graspf[int(l*start):int(l*end)]
        self.depth_files = depthf[int(l*start):int(l*end)]
        self.rgb_files = rgbf[int(l*start):int(l*end)]
        print('ALL IMAGE LISTS ARE CREATED!')
        print('Depth Image List:{}, Type: {}, Shape: {}'.format(self.depth_files, type(self.depth_files), len(self.depth_files)))


    def _get_crop_attrs(self, idx):
        # TODO: improve function to get center - not hardcoded
        center = (240,320)
        left = max(0, min(center[1] - self.output_size // 2, 640 - self.output_size))
        top = max(0, min(center[0] - self.output_size // 2, 480 - self.output_size))
        return center, left, top

    # this is the important function which loads the depth image and performs preprocessing
    # wie wird get_depth aufgerufen??????? -> aus GraspDatasetBase Klasse (grasp_data.py), von der alle Dataset-Klassen (ycb, jaquard, cornell) erben

    def get_depth(self, idx, rot=0, zoom=1.0):
        print('INDEX: {}'.format(self.depth_files[idx]))

        # load the depth image
        depth_img = image.DepthImage.from_npy(self.depth_files[idx])
        print('DEPTH IMAGE: Type: {}, Shape: {}'.format(type(depth_img), depth_img.shape))
        
        # we have to crop because image is not (a x a) but (480 x 640)
        center, left, top = self._get_crop_attrs(idx)
        depth_img.rotate(rot, center)
        depth_img.crop((top, left), (min(480, top + self.output_size), min(640, left + self.output_size)))
        depth_img.normalise()
        depth_img.zoom(zoom)
        depth_img.resize((self.output_size, self.output_size))
        return depth_img.img

    def get_rgb(self, idx, rot=0, zoom=1.0, normalise=True):
        rgb_img = image.Image.from_file(self.rgb_files[idx])
        center, left, top = self._get_crop_attrs(idx)
        rgb_img.rotate(rot, center)
        rgb_img.crop((top, left), (min(480, top + self.output_size), min(640, left + self.output_size)))
        rgb_img.zoom(zoom)
        rgb_img.resize((self.output_size, self.output_size))
        if normalise:
            rgb_img.normalise()
            rgb_img.img = rgb_img.img.transpose((2, 0, 1))
        return rgb_img.img
