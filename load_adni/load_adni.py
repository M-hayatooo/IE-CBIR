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
    strength={
        "1.5",
        "3.0",
        "1.494",
        "2.89362",
    },
    size="half",
    unique=True,
    mni=False,
    dryrun=False,
):
    adni1 = "/data2/radiology_datas/clean3/meta/json/ADNI1.json"
    adni1_json = json.loads(Path(adni1).read_text())

    matching_images = []
    pid_list = []
    for subject in adni1_json:
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
                nib.squeeze_image(nib.as_closest_canonical(nib.load(load_path)))
                .get_fdata()
                .astype("float32")
            )
    return matching_images


def load_adni2(
    classes={
        "CN",
        "AD",
        "MCI",
        # "nan",
    },
    strength={
        "1.5",
        "3.0",
    },
    size="half",
    unique=True,
    mni=False,
    dryrun=False,
):
    adni2 = "/data2/radiology_datas/clean3/meta/json/ADNI2.json"
    adni2_json = json.loads(Path(adni2).read_text())

    matching_images = []
    pid_list = []
    for subject in adni2_json:
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
                nib.squeeze_image(nib.as_closest_canonical(nib.load(load_path)))
                .get_fdata()
                .astype("float32")
            )
    return matching_images


def load_adni3(
    classes={
        "CN",
        "AD",
        "MCI",
        # "nan",
    },
    size="half",
    unique=True,
    mni=False,
    dryrun=False,
):
    adni3 = "/data2/radiology_datas/clean3/meta/json/ADNI3.json"
    adni3_json = json.loads(Path(adni3).read_text())

    matching_images = []
    pid_list = []
    for subject in adni3_json:
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
                nib.squeeze_image(nib.as_closest_canonical(nib.load(load_path)))
                .get_fdata()
                .astype("float32")
            )
    return matching_images
