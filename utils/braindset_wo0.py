from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
from torch.utils.data import Dataset

CLASS_MAP = {"CN": 0, "AD": 1, "MCI":2}

class BrainDataset(Dataset):
    def __init__(self, voxels, labels, transform=None, phase="train"):
        self.voxels = [self._preprocess(data) for data in voxels]
        self.labels = labels
        self.phase = phase
        self.transform = transform

    def __len__(self):
        return len(self.voxels)
    
    def __getitem__(self, index):
        voxel = self.voxels[index]
        label = self.labels[index]
        if self.transform:
            voxel = self.transform(voxel)
        return voxel, label
    
    def _preprocess(self, voxel):
        # cut_range = 2          # 輝度0から、平均＋標準偏差×2倍の位置でclip
        nonzero = voxel[voxel>0.0] # 平均と標準偏差の計算に輝度0は含めない
        voxel = np.clip(voxel, 0, np.mean(nonzero) + 2.0*np.std(nonzero))
        voxel = normalize(voxel, np.min(voxel), np.max(voxel))
        voxel = voxel[np.newaxis, ]
        return voxel.astype('f')
    
    def __call__(self, index):
        return self.__getitem__(index)
    
    # def before_preprocess(self, voxel):
    #     cut_range = 4
    #     voxel = np.clip(voxel, 0, cut_range * np.std(voxel))
    #     voxel = normalize(voxel, np.min(voxel), np.max(voxel))
    #     voxel = voxel[np.newaxis, ]
    #     return voxel.astype('f')

def normalize(voxel: np.ndarray, floor: int, ceil: int) -> np.ndarray:
    return (voxel - floor) / (ceil - floor)
