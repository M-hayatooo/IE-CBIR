import json
from pathlib import Path

import nibabel as nib
import numpy as np
from tqdm import tqdm


def load_oasis1(
    classes={
        "CN",
        "probable AD",
        # "nan",
    },
    size="half", unique=True, mni=False, dryrun=False,
):
    oasis1 = "/data2/radiology_datas/clean3/meta/json/OASIS1.json"
    oasis1_json = json.loads(Path(oasis1).read_text())

    matching_images = []
    pid_list = []
    for subject in oasis1_json:
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


def load_oasis2(
    classes={
        "CN",
        "AD",
        # "nan",
    },
    size="half",
    unique=True,
    mni=False,
    dryrun=False,
):
    oasis2 = "/data2/radiology_datas/clean3/meta/json/OASIS2.json"
    oasis2_json = json.loads(Path(oasis2).read_text())

    matching_images = []
    pid_list = []
    for subject in oasis2_json:
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


def load_oasis3(
    classes={
        "CN",
        "AD",
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
    oasis3 = "/data2/radiology_datas/clean3/meta/json/OASIS3.json"
    oasis3_json = json.loads(Path(oasis3).read_text())

    matching_images = []
    pid_list = []
    for subject in oasis3_json:
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


def load_oasis4(
    classes={
        "CN",
        "MCI",
        "AD",
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
    oasis4 = "/home/macky/workspace3/OMAP-1/notebook/json/OASIS4.json"
    oasis4_json = json.loads(Path(oasis4).read_text())

    matching_images = []
    pid_list = []
    for subject in oasis4_json:
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