from .dataset import MiddleburyDataset, SceneFlowDataset, KITTIDataset, DrivingStereoDataset

__datasets__ = {
    "middlebury": MiddleburyDataset,
    "sceneflow": SceneFlowDataset,
    "kitti": KITTIDataset,
    "drivingstereo": DrivingStereoDataset,
}
