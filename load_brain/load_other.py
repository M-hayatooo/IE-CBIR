import json
from pathlib import Path

import nibabel as nib
import numpy as np
from tqdm import tqdm


def load_other(
    datasets={
        "CC359",
        "LPBA40",
        "NFBS",
    },
    size="half",
    mni=False,
    dryrun=False,
):
    all_subjects = []
    for dataset in datasets:
        if dataset == "CC359":
            data = "/data2/radiology_datas/clean3/meta/json/CC359.json"
        if dataset == "LPBA40":
            data = "/data2/radiology_datas/clean3/meta/json/LPBA40.json"
        if dataset == "NFBS":
            data = "/data2/radiology_datas/clean3/meta/json/NFBS.json"
        data_json = json.loads(Path(data).read_text())
        all_subjects += data_json

    for subject in tqdm(all_subjects):
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
    return all_subjects
