from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
from torch.utils.data import Dataset

CLASS_MAP = {"CN": 0, "AD": 1, "MCI":2}

class BrainDataset(Dataset):
    def __init__(self, voxels, labels, transform=None, phase="train"):
        # self.voxels = voxels
        # self.voxels = self._preprocess(voxels)
        # with ThreadPoolExecutor(max_workers=32) as e:
        #     for data in voxels:
        #         temp = e.submit(self._preprocess(data))
        #  
        self.voxels = [self._preprocess(data) for data in voxels]
        # ----------------------------------------------------------
        # self.voxels = [self._preprocess(data["voxel"]) for data in voxels]
        # self.voxels=[self._preprocess(v) for v in voxels] # self.voxels=[self._preprocess(data["voxel"]) for data in self.data]
        # self.voxels = temp
        self.labels = labels
        self.phase = phase
        self.transform = transform
    def __len__(self):
        return len(self.voxels)
    def __getitem__(self, index):
        # x_tensor = torch.from_numpy(x_numpy).clone()
        # voxel = torch.from_numpy(self.voxels[index]).to(self.device)
        # label = torch.from_numpy(self.labels[index]).to(self.device)
        voxel = self.voxels[index]
        label = self.labels[index]
        # (voxel = self._preprocess(voxel))
        if self.transform:
            voxel = self.transform(voxel)
            # voxel = self.transform(voxel, self.phase)
        return voxel, label
    def _preprocess(self, voxel):
        cut_range = 4
        voxel = np.clip(voxel, 0, cut_range * np.std(voxel))
        voxel = normalize(voxel, np.min(voxel), np.max(voxel))
        voxel = voxel[np.newaxis, ]
        return voxel.astype('f')
    def __call__(self, index):
        return self.__getitem__(index)

#                         ""min → floor""  "" max → ceil ""
def normalize(voxel: np.ndarray, floor: int, ceil: int) -> np.ndarray:
    return (voxel - floor) / (ceil - floor)
    # voxel - min
    # ―――――――――――
    # max   - min
