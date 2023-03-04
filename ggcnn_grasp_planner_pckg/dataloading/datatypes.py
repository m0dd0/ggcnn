from dataclasses import dataclass
from abc import ABC

from nptyping import NDArray, Shape, Float, Int


@dataclass
class DatasetSample(ABC):
    name: str


@dataclass
class YCBSimulationDataSample(DatasetSample):
    name: str
    rgb: NDArray[Shape["H, W, 3"], Int]
    depth: NDArray[Shape["H, W"], Float]
    segmentation: NDArray[Shape["H, W"], Int]
    points: NDArray[Shape["N, 3"], Float]
    points_color: NDArray[Shape["N, 3"], Int]
    points_segmented: NDArray[Shape["N, 3"], Float]
    points_segmented_color: NDArray[Shape["N, 3"], Int]
    cam_intrinsics: NDArray[Shape["3, 3"], Float]
    cam_pos: NDArray[Shape["3"], Float]
    cam_rot: NDArray[Shape["3, 3"], Float]
