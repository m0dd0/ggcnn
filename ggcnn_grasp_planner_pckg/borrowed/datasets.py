from typing import Callable, List
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from .datatypes import YCBSimulationDataSample


class YCBSimulationData:
    def __init__(
        self, root_dir: Path, invalid_objs: List[str] = None, transform: Callable = None
    ):
        self.root_dir = Path(root_dir).expanduser()
        self.transform = transform

        self.invalid_objs = invalid_objs
        if self.invalid_objs is None:
            self.invalid_objs = []

        self.all_sample_names = sorted(
            [
                p.parts[-1]
                for p in self.root_dir.iterdir()
                if p.suffix == ".npz" and p.stem not in self.invalid_objs
            ]
        )

    def __len__(self):
        return len(self.all_sample_names)

    def __getitem__(self, index: int) -> YCBSimulationDataSample:
        if index >= len(self):
            raise IndexError(
                f"Index {index} out of range for dataset with length {len(self)}"
            )

        sample_name = self.all_sample_names[index]
        sample_path = self.root_dir / sample_name

        simulation_data = np.load(sample_path)

        sample = YCBSimulationDataSample(
            rgb=simulation_data["rgb_img"],
            depth=simulation_data["depth_img"],
            points=simulation_data["point_cloud"][0],
            points_color=(simulation_data["point_cloud"][1] * 255).astype(np.uint8),
            points_segmented=simulation_data["point_cloud_seg"][0],
            points_segmented_color=(simulation_data["point_cloud_seg"][1] * 255).astype(
                np.uint8
            ),
            segmentation=simulation_data["seg_img"].astype("uint8"),
            cam_intrinsics=simulation_data["cam_intrinsics"],
            cam_pos=simulation_data["cam_pos"],
            cam_rot=Rotation.from_quat(
                simulation_data["cam_quat"][[1, 2, 3, 0]]
            ).as_matrix(),
            name=sample_name.split(".")[0],
        )

        if self.transform is not None:
            sample = self.transform(sample)

        return sample
