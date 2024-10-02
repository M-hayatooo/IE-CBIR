import random

import load_adni.load_adni as load_adni
import numpy as np
import torch
import torchio as tio
from sklearn.model_selection import StratifiedGroupKFold, train_test_split
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.braindset_wo0 import BrainDataset

SEED_VALUE = 103
CLASS_MAP = {"CN": 0, "AD": 1, "MCI":2}

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def load_dataloader(n_train_rate, batch_size, augment_flag, index_num, class_set):
    # data = load_data(kinds=["ADNI2", "ADNI2-2"], classes=["CN", "AD", "EMCI", "LMCI", "SMC", "MCI"], unique=False, blacklist=True)
    data = load_adni.load_adni2(
        classes=class_set, strength={"3.0"},
        size="half", unique=False,
        # mni=False, dryrun=False
    )
    # data += load_adni.load_adni3(
    #     classes={"CN", "AD", "MCI", "EMCI", "LMCI", "SMC"},
    #     size="half", unique=False, dryrun=False)
    pids = []
    # before_voxels = np.zeros((len(data), 80, 112, 80))
    voxels = np.zeros((len(data), 80, 112, 80))
    labels = np.zeros(len(data))
    for i in tqdm(range(len(data))):
        pids.append(data[i]["pid"])
        voxels[i] = data[i]["voxel"]
        # voxels[i] = before_voxels[i,:,8:104,:] # 80x112x80 -> 80x96x80
        labels[i] = CLASS_MAP[data[i]["class"]] # skull strippingの時はここを変更した。

    print(f"data len = {len(data)}")
    pids = np.array(pids)

#   split index  を指定
    split_index = index_num # 0~4
    sgk = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED_VALUE)
    tid, vid = list(sgk.split(voxels, y=labels, groups=pids))[split_index]
#    tid, vid = list(gss.split(voxels, groups=pids))[0]

    train_voxels = voxels[tid]
    train_labels = labels[tid]
    val_voxels = voxels[vid]
    val_labels = labels[vid]

    print(f"fold number={split_index}   Augment={augment_flag}", end="  ")
    if augment_flag: # TrueならAugmentation実行, Falseなら実行しない
        spatial_transforms = {# soft-introだと30°は回転の影響が大き過ぎるかも
            tio.transforms.RandomAffine(degrees=(5.0, 0.0, 0.0)): 0.340,
            tio.transforms.RandomAffine(degrees=(0.0, 5.0, 0.0)): 0.330,
            tio.transforms.RandomAffine(degrees=(0.0, 0.0, 5.0)): 0.330,
        }
        transform = tio.Compose([
            tio.OneOf(spatial_transforms, p=0.6),
        ])
        train_dataset = BrainDataset(train_voxels, train_labels, transform=transform, phase="train")
        val_dataset = BrainDataset(val_voxels, val_labels, transform=None, phase="val")
    else:
        train_dataset = BrainDataset(train_voxels, train_labels, transform=None, phase="train")
        val_dataset = BrainDataset(val_voxels, val_labels, transform=None, phase="val")


    g = torch.Generator()
    g.manual_seed(SEED_VALUE)

    print(f"batch size:{batch_size}", end="  ")
    train_dataloader=DataLoader(train_dataset, batch_size=batch_size, num_workers=4,
                                pin_memory=True, shuffle=True, worker_init_fn=seed_worker, generator=g)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, num_workers=4,
                                pin_memory=True, shuffle=False, worker_init_fn=seed_worker, generator=g)

    return train_dataloader, val_dataloader, train_dataset, val_dataset
