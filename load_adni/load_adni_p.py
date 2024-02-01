import json
from pathlib import Path

import nibabel as nib
import numpy as np
from tqdm import tqdm


def load_adni1(
    classes={
        "CN",
        "AD",
        "MCI",
        # "nan",
    },
    strength={"1.5","3.0","1.494","2.89362",},
    size="half", unique=True, mni=False, dryrun=False,
):
    adni1 = "/data2/radiology_datas/clean2/json/ADNI1_PRE.json"
    adni1_json = json.loads(Path(adni1).read_text())
    all_subjects = adni1_json

    matching_images = []
    pid_list = []
    for subject in all_subjects:
        if subject["class"] not in classes:
            continue
        if subject["strength"] not in strength:
            continue
        if (unique == True) and subject["pid"] in pid_list:
            continue

        pid_list.append(subject["pid"])
        matching_images.append(subject)

    for subject in tqdm(matching_images):
        if not dryrun:
            if size == "half" and mni == False:
                load_path = subject["path_half"]
            elif size == "full" and mni == False:
                load_path = subject["path_full"]
            elif size == "half" and mni == True:
                load_path = subject["path_half_mni"]
            elif size == "full" and mni == True:
                load_path = subject["path_full_mni"]
            subject["voxel"] = (
                nib.load(load_path).get_fdata()[:, :, :, 0].astype("float32")
            )
    return matching_images

def load_adni2(
    classes={
        "CN",
        "AD",
        "MCI",
        # "nan",
    },
    strength={"1.5","3.0"}, size="half", unique=True, mni=False, dryrun=False,
):
    adni2 = "/data2/radiology_datas/clean2/json/ADNI2_PRE.json"
    adni2_json = json.loads(Path(adni2).read_text())
    all_subjects = adni2_json

    matching_images = []
    pid_list = []
    for subject in all_subjects:
        if subject["class"] not in classes:
            continue
        if subject["strength"] not in strength:
            continue
        if (unique == True) and subject["pid"] in pid_list:
            continue

        pid_list.append(subject["pid"])
        matching_images.append(subject)

    for subject in tqdm(matching_images):
        if not dryrun:
            if size == "half" and mni == False:
                load_path = subject["path_half"]
            elif size == "full" and mni == False:
                load_path = subject["path_full"]
            elif size == "half" and mni == True:
                load_path = subject["path_half_mni"]
            elif size == "full" and mni == True:
                load_path = subject["path_full_mni"]
            subject["voxel"] = (
                nib.load(load_path).get_fdata()[:, :, :, 0].astype("float32")
            )
    return matching_images


def load_adni3(
    classes={
        "CN",
        "AD",
        "MCI",
        # "nan",
    },
    size="half", unique=True,
    mni=False,
    dryrun=False,
):
    adni3 = "/data2/radiology_datas/clean2/json/ADNI3_MPRAGE.json"
    adni3_json = json.loads(Path(adni3).read_text())
    all_subjects = adni3_json

    matching_images = []
    pid_list = []
    for subject in all_subjects:
        if subject["class"] not in classes:
            continue
        if (unique == True) and subject["pid"] in pid_list:
            continue

        pid_list.append(subject["pid"])
        matching_images.append(subject)

    for subject in tqdm(matching_images):
        if not dryrun:
            if size == "half" and mni == False:
                load_path = subject["path_half"]
            elif size == "full" and mni == False:
                load_path = subject["path_full"]
            elif size == "half" and mni == True:
                load_path = subject["path_half_mni"]
            elif size == "full" and mni == True:
                load_path = subject["path_full_mni"]
            subject["voxel"] = (
                nib.load(load_path).get_fdata()[:, :, :, 0].astype("float32")
            )
    return matching_images
