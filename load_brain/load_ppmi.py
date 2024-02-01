import json
from pathlib import Path

import nibabel as nib
import numpy as np
from tqdm import tqdm


def load_ppmi(
    classes={
        "CN",
        "PD",
    },
    size="half",
    unique=True,
    mni=False,
    dryrun=False,
):
    ppmi = "/data2/radiology_datas/clean3/meta/json/PPMI.json"
    ppmi_json = json.loads(Path(ppmi).read_text())

    matching_images = []
    pid_list = []
    for subject in ppmi_json:
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
