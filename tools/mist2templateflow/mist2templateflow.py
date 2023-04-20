"""
Downloading NeuroImaging datasets: atlas datasets
"""
import argparse
from pathlib import Path
import shutil
import os
import numpy as np
import pandas as pd

from sklearn.utils import Bunch

from nilearn.datasets import fetch_atlas_basc_multiscale_2015
from nilearn.datasets.utils import _get_dataset_dir, _fetch_files
from nilearn.image import load_img


# download the release to data/raw
DOWNLOAD_URL = "https://figshare.com/ndownloader/files/9811081"
TEMPLATE = {
    "BASC": {"asym": "MNI152NLin2009bAsym", "sym": "MNI152NLin2009bSym"},
    "MIST": "MNI152NLin2009bSym",
}
DESCRIPTIONS = {
    "BASC": [7, 12, 20, 36, 64, 122, 197, 325, 444],
    "MIST": [
        7,
        12,
        20,
        36,
        64,
        122,
        197,
        325,
        444,
        "ROI",
        "ATOM",
        "Hierarchy",
    ],
}

INPUT_DIR = "./data/raw"
OUTPUT_DIR = "./data/processed"


def get_parser():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="",
        epilog="""
    Convert MIST to TemplateFlow competible convention
    """,
    )
    parser.add_argument(
        "-o",
        "--output",
        required=False,
        help="" "Output directory.",
    )
    parser.add_argument(
        "-d",
        action="store_true",
        help="" "Delete original data.",
    )
    return parser


def fetch_atlas_basc(
    dimension, tpl_ver, data_dir=None, resume=True, verbose=1
):
    """Get the BASC atlas (the base version of MIST)."""
    dataset_name = "original_BASC"
    data_dir = _get_dataset_dir(
        dataset_name, data_dir=data_dir, verbose=verbose
    )
    filename = fetch_atlas_basc_multiscale_2015(
        tpl_ver, data_dir=data_dir, resume=resume, verbose=verbose
    )[f"scale{dimension:03d}"]

    params = dict(zip(["maps"], [filename]))
    tpl = TEMPLATE["BASC"][tpl_ver]
    tpf = convert_templateflow(tpl, "BASC", dimension)
    params.update(tpf)
    return Bunch(**params)


def fetch_atlas_mist(
    dimension, data_dir=None, url=None, resume=True, verbose=1
):
    """Downloads MIST from https://figshare.com/ndownloader/files/9811081"""
    if dimension not in DESCRIPTIONS["MIST"]:
        raise ValueError(f"{dimension} doesn't exist.")
    if url is None:
        url = DOWNLOAD_URL

    opts = {"uncompress": True}

    dataset_name = "original_MIST2019"
    data_dir = _get_dataset_dir(
        dataset_name, data_dir=data_dir, verbose=verbose
    )
    if dimension == "Hierarchy":
        filenames = [
            (
                os.path.join(
                    "Release", "Hierarchy", "MIST_PARCEL_ORDER_ROI.csv"
                ),
                url,
                opts,
            ),
            (
                os.path.join("Release", "Hierarchy", "MIST_PARCEL_ORDER.csv"),
                url,
                opts,
            ),
        ]
        keys = ["Hierarchy_ROI", "Hierarchy"]

    elif dimension == "ATOM":
        filenames = [
            (
                os.path.join("Release", "Parcellations", "MIST_ATOM.nii.gz"),
                url,
                opts,
            ),
        ]
        keys = ["maps"]
    else:
        filenames = [
            (
                os.path.join(
                    "Release", "Parcellations", f"MIST_{dimension}.nii.gz"
                ),
                url,
                opts,
            ),
            (
                os.path.join(
                    "Release", "Parcel_Information", f"MIST_{dimension}.csv"
                ),
                url,
                opts,
            ),
        ]
        keys = ["maps", "labels"]

    files_ = _fetch_files(data_dir, filenames, resume=resume, verbose=verbose)
    params = dict(zip(keys, files_))
    if dimension == "ATOM":
        atom_img = load_img(files_[0])
        n_atoms = np.unique(atom_img.dataobj).shape[-1]
        params["labels"] = list(range(1, n_atoms))

    tpf = convert_templateflow(TEMPLATE["MIST"], "MIST", dimension)
    params.update(tpf)
    return Bunch(**params)


def convert_templateflow(template, atlas, desc):
    folder_name = f"tpl-{template}"
    if desc != "Hierarchy":
        basenames = f"tpl-{template}_res-03_atlas-{atlas}_desc-{desc}_dseg"
        keys = ["tpf_maps", "tpf_labels"]
        filenames = [
            os.path.join(folder_name, f"{basenames}.nii.gz"),
            os.path.join(folder_name, f"{basenames}.tsv"),
        ]
    else:
        descs = ["ParcelHierarchyROI", "ParcelHierarchy"]
        filenames = [
            os.path.join(
                folder_name,
                f"tpl-{template}_res-03_atlas-{atlas}_desc-{desc}_dseg.tsv",
            )
            for desc in descs
        ]
        keys = [f"tpf_{desc}" for desc in descs]
    return dict(zip(keys, filenames))


def convert_basc(output_dir, input_dir):
    for desc in DESCRIPTIONS["BASC"]:
        for tpl_ver in ["sym", "asym"]:
            dataset = fetch_atlas_basc(desc, tpl_ver, data_dir=input_dir)
            nii = Path(dataset["maps"])
            output_file = Path(output_dir) / dataset["tpf_maps"]

            if not output_file.parent.is_dir():
                output_file.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(nii, output_file)
        print(
            "Convert data and save to "
            f"{output_dir}/tpl-{TEMPLATE['BASC'][tpl_ver]}"
        )


def convert_mist(output_dir, input_dir):
    for desc in DESCRIPTIONS["MIST"]:
        dataset = fetch_atlas_mist(desc, data_dir=input_dir)
        if desc != "Hierarchy":
            nii = Path(dataset["maps"])
            output_file = Path(output_dir) / dataset["tpf_maps"]

            if not output_file.parent.is_dir():
                output_file.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(nii, output_file)

            if desc == "ATOM":
                labels = pd.DataFrame(dataset["labels"], columns=["roi"])
            else:
                labels = pd.read_csv(dataset["labels"], sep=";")
            labels.to_csv(
                os.path.join(output_dir, dataset["tpf_labels"]),
                index=False,
                sep="\t",
            )
        else:
            for label, tpf in zip(
                ["Hierarchy_ROI", "Hierarchy"],
                ["ParcelHierarchyROI", "ParcelHierarchy"],
            ):
                df = pd.read_csv(dataset[label])
                df.to_csv(
                    os.path.join(output_dir, dataset[f"tpf_{tpf}"]),
                    index=False,
                    sep="\t",
                )
    print(f"Convert data and save to {output_dir}/tpl-{TEMPLATE['MIST']}")


def main():
    args = get_parser().parse_args()

    if not args.output:
        output_dir = OUTPUT_DIR
        input_dir = INPUT_DIR
    else:
        output_dir = args.output
        input_dir = args.output

    convert_mist(output_dir, input_dir)
    convert_basc(output_dir, input_dir)

    if args.d:
        print("Delete original data")
        shutil.rmtree(Path(input_dir) / "original_MIST2019")
        shutil.rmtree(Path(input_dir) / "original_BASC")


if __name__ == "__main__":
    main()
