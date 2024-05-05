import os
import subprocess
import argparse
import sqlite3
from PIL import Image
import cv2
from tqdm import tqdm
from PIL import Image, ExifTags
import numpy as np


# local imports
from colmap.colmap import database
from utils import database_handler

# lightglue imports
import torch
from lightglue import LightGlue, SuperPoint, DISK, SIFT
from lightglue.utils import load_image, rbd

torch.set_grad_enabled(False)


# --------------------------------------------------------------------------------#
def get_minimum_dimensions(image_folder):
    min_width, min_height = float("inf"), float("inf")
    for image_name in tqdm(os.listdir(image_folder)):
        if image_name.lower().endswith(
            (".png", ".jpg", ".jpeg", ".bmp", ".gif", ".JPG")
        ):
            image_path = os.path.join(image_folder, image_name)
            with Image.open(image_path) as img:
                width, height = img.size
                min_width, min_height = min(min_width, width), min(min_height, height)
    return min_width, min_height


def crop_images_to_min_size(image_folder, min_width, min_height, output_dir):
    for idx, image_name in tqdm(
        enumerate(os.listdir(image_folder)), total=len(os.listdir(image_folder))
    ):
        if image_name.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".gif")):
            image_path = os.path.join(image_folder, image_name)
            output_image_path = os.path.join(output_dir, f"{idx}.jpg")
            image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
            # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            image = image[:min_height, :min_width, :]
            cv2.imwrite(output_image_path, image)


def preprocess_images(image_folder, output_dir):
    # fix the image size to make it uniform
    min_width, min_height = get_minimum_dimensions(image_folder)
    os.makedirs(output_dir, exist_ok=True)
    crop_images_to_min_size(image_folder, min_width, min_height, output_dir)


# --------------------------------------------------------------------------------#
def get_focal(image_path, err_on_default=False):
    image = Image.open(image_path)
    max_size = max(image.size)

    exif = image.getexif()
    focal = None
    if exif is not None:
        focal_35mm = None
        # https://github.com/colmap/colmap/blob/d3a29e203ab69e91eda938d6e56e1c7339d62a99/src/util/bitmap.cc#L299
        for tag, value in exif.items():
            focal_35mm = None
            if ExifTags.TAGS.get(tag, None) == "FocalLengthIn35mmFilm":
                focal_35mm = float(value)
                break

        if focal_35mm is not None:
            focal = focal_35mm / 35.0 * max_size

    if focal is None:
        if err_on_default:
            raise RuntimeError("Failed to find focal length")

        # failed to find it in exif, use prior
        FOCAL_PRIOR = 1.2
        focal = FOCAL_PRIOR * max_size

    return focal


def create_camera(db, image_path):
    image = Image.open(image_path)
    width, height = image.size

    focal = get_focal(image_path)

    model = 0  # simple pinhole
    param_arr = np.array([focal, width / 2, height / 2])

    return db.add_camera(model, width, height, param_arr)


def create_database(
    workspace_dir: str, database_file: str, features: dict, matches: dict
):
    # create a database
    db = database.COLMAPDatabase.connect(database_file)
    db.create_tables()
    # create camera
    random_image = list(features.keys())[0]
    random_image_path = os.path.join(workspace_dir, "preprocessed_images", random_image)
    camera_id = create_camera(db, random_image_path)
    # add images
    fname_to_id = {}
    for img_file in features:
        image_id = db.add_image(img_file, camera_id=camera_id)
        fname_to_id[img_file] = image_id
    db.commit()
    # add keypoints
    for img_file, feature in features.items():
        img_id = fname_to_id[img_file]
        keypoints = feature["keypoints"][0].cpu().numpy()
        db.add_keypoints(img_id, keypoints)
    db.commit()
    # add matches
    for (img_file1, img_file2), match in matches.items():
        img_id1, img_id2 = fname_to_id[img_file1], fname_to_id[img_file2]
        matches = match["matches"][0].cpu().numpy()
        db.add_matches(img_id1, img_id2, matches)
    db.commit()

    db.close()


# --------------------------------------------------------------------------------#
"""
TODO: MERGING DATABASES 


"""


def database_merger_handler(
    workspace_dir: str, methods: list = ["disk", "sift", "superpoint"]
):
    sift_database_path = f"{workspace_dir}/sift/database.db"
    disk_database_path = f"{workspace_dir}/disk/database.db"
    superpoint_database_path = f"{workspace_dir}/superpoint/database.db"

    # step 1: merge sift and disk
    sift_disk_merged_database_path = f"{workspace_dir}/sift_disk_merged_database.db"
    db = database.COLMAPDatabase.connect(sift_disk_merged_database_path)
    db.create_tables()
    db.close()
    ### copy the existing tables database
    database_handler.copy_colmap_database(
        sift_database_path, sift_disk_merged_database_path
    )

    ### merge the keypoints
    database_handler.merge_keypoint(
        sift_database_path,
        disk_database_path,
        sift_disk_merged_database_path,
    )
    ### merge the matches
    database_handler.merge_matches(
        sift_database_path,
        disk_database_path,
        sift_disk_merged_database_path,
    )

    # step 2: merge sift_disk and superpoint
    sift_disk_superpoint_merged_database_path = f"{workspace_dir}/merged_database.db"
    db = database.COLMAPDatabase.connect(sift_disk_superpoint_merged_database_path)
    db.create_tables()
    db.close()
    ### copy the existing tables database
    database_handler.copy_colmap_database(
        sift_disk_merged_database_path, sift_disk_superpoint_merged_database_path
    )

    ### merge the keypoints
    database_handler.merge_keypoint(
        sift_disk_merged_database_path,
        superpoint_database_path,
        sift_disk_superpoint_merged_database_path,
    )

    ### merge the matches

    database_handler.merge_matches(
        sift_disk_merged_database_path,
        superpoint_database_path,
        sift_disk_superpoint_merged_database_path,
    )

    # close the databases


# --------------------------------------------------------------------------------#
"""

TODO:

    1. DISK 
        - EXTRACT FEATURES
        - MATCH FEATURES
        - CREATE DATABASE
    2. SIFT
        - EXTRACT FEATURES
        - MATCH FEATURES
        - CREATE DATABASE
    3. DoGHardNet
        - EXTRACT FEATURES
        - MATCH FEATURES
        - CREATE DATABASE
    4. SuperPoint
        - EXTRACT FEATURES
        - MATCH FEATURES
        - CREATE DATABASE

"""
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
print("======================================")


def disk_handler(workspace_dir: str):
    disk_dir = f"{workspace_dir}/disk"
    os.makedirs(disk_dir, exist_ok=True)

    image_dir = f"{workspace_dir}/preprocessed_images"
    extractor = DISK().eval().to(device)
    matcher = LightGlue(features="disk").eval().to(device)
    # 1. EXTRACT FEATURES
    features = {}
    for img_file in tqdm(os.listdir(image_dir), desc="Extracting DISK Features :: "):
        img = load_image(os.path.join(image_dir, img_file)).to(device)
        feature = extractor.extract(img)
        features[img_file] = feature

    # 2. MATCH FEATURES
    matches = {}
    for img_file1 in tqdm(os.listdir(image_dir), desc="Matching Features :: "):
        for img_file2 in os.listdir(image_dir):
            if (
                img_file1 != img_file2
                and (img_file1, img_file2) not in matches
                and (img_file2, img_file1) not in matches
            ):
                match = matcher(
                    {"image0": features[img_file1], "image1": features[img_file2]}
                )
                matches[(img_file1, img_file2)] = match
    # 3. CREATE DATABASE
    database_file = f"{disk_dir}/database.db"
    # delete database file if it exists
    if os.path.exists(database_file):
        subprocess.run(f"rm -rf {database_file}", shell=True)
    create_database(
        workspace_dir=workspace_dir,
        database_file=database_file,
        features=features,
        matches=matches,
    )


def sift_handler(workspace_dir: str):
    colmap_dir = f"{workspace_dir}/sift"
    os.makedirs(colmap_dir, exist_ok=True)

    image_dir = f"{workspace_dir}/preprocessed_images"
    extractor = SIFT().eval().to(device)
    matcher = LightGlue(features="sift").eval().to(device)

    # 1. EXTRACT FEATURES
    features = {}
    for img_file in tqdm(os.listdir(image_dir), desc="Extracting SIFT Features :: "):
        img = load_image(os.path.join(image_dir, img_file)).to(device)
        feature = extractor.extract(img)
        features[img_file] = feature

    # 2. MATCH FEATURES
    matches = {}
    for img_file1 in tqdm(os.listdir(image_dir), desc="Matching Features :: "):
        for img_file2 in os.listdir(image_dir):
            if (
                img_file1 != img_file2
                and (img_file1, img_file2) not in matches
                and (img_file2, img_file1) not in matches
            ):
                match = matcher(
                    {"image0": features[img_file1], "image1": features[img_file2]}
                )
                matches[(img_file1, img_file2)] = match

    # 3. CREATE DATABASE
    database_file = f"{colmap_dir}/database.db"
    # delete database file if it exists
    if os.path.exists(database_file):
        subprocess.run(f"rm -rf {database_file}", shell=True)
    create_database(
        workspace_dir=workspace_dir,
        database_file=database_file,
        features=features,
        matches=matches,
    )


def doghardnet_handler(workspace_dir: str):
    doghardnet_dir = f"{workspace_dir}/doghardnet"
    os.makedirs(doghardnet_dir, exist_ok=True)

    image_dir = f"{workspace_dir}/preprocessed_images"
    extractor = DISK().eval().to(device)
    matcher = LightGlue(features="doghardnet").eval().to(device)

    # 1. EXTRACT FEATURES
    features = {}
    for img_file in tqdm(
        os.listdir(image_dir), desc="Extracting DoGHardNet Features :: "
    ):
        img = load_image(os.path.join(image_dir, img_file)).to(device)
        feature = extractor.extract(img)
        features[img_file] = feature

    # 2. MATCH FEATURES
    matches = {}
    for img_file1 in tqdm(os.listdir(image_dir), desc="Matching Features :: "):
        for img_file2 in os.listdir(image_dir):
            if (
                img_file1 != img_file2
                and (img_file1, img_file2) not in matches
                and (img_file2, img_file1) not in matches
            ):
                match = matcher(
                    {"image0": features[img_file1], "image1": features[img_file2]}
                )
                matches[(img_file1, img_file2)] = match

    # 3. CREATE DATABASE
    database_file = f"{doghardnet_dir}/database.db"
    # delete database file if it exists
    if os.path.exists(database_file):
        subprocess.run(f"rm -rf {database_file}", shell=True)

    create_database(
        workspace_dir=workspace_dir,
        database_file=database_file,
        features=features,
        matches=matches,
    )


def superpoint_handler(workspace_dir: str):
    superpoint_dir = f"{workspace_dir}/superpoint"
    os.makedirs(superpoint_dir, exist_ok=True)

    image_dir = f"{workspace_dir}/preprocessed_images"
    extractor = SuperPoint().eval().to(device)
    matcher = LightGlue(features="superpoint").eval().to(device)

    # 1. EXTRACT FEATURES
    features = {}
    for img_file in tqdm(
        os.listdir(image_dir), desc="Extracting SuperPoint Features :: "
    ):
        img = load_image(os.path.join(image_dir, img_file)).to(device)
        feature = extractor.extract(img)
        features[img_file] = feature

    # 2. MATCH FEATURES
    matches = {}
    for img_file1 in tqdm(os.listdir(image_dir), desc="Matching Features :: "):
        for img_file2 in os.listdir(image_dir):
            if (
                img_file1 != img_file2
                and (img_file1, img_file2) not in matches
                and (img_file2, img_file1) not in matches
            ):
                match = matcher(
                    {"image0": features[img_file1], "image1": features[img_file2]}
                )
                matches[(img_file1, img_file2)] = match

    # 3. CREATE DATABASE
    database_file = f"{superpoint_dir}/database.db"
    # delete database file if it exists
    if os.path.exists(database_file):
        subprocess.run(f"rm -rf {database_file}", shell=True)

    create_database(
        workspace_dir=workspace_dir,
        database_file=database_file,
        features=features,
        matches=matches,
    )


# --------------------------------------------------------------------------------#
def main(workspace_dir: str):
    # 1. PREPROCESS THE IMAGES
    preprocessed_image_dir = f"{workspace_dir}/preprocessed_images"
    # delete if the directory exists
    if not os.path.exists(preprocessed_image_dir):
        # subprocess.run(f"rm -rf {preprocessed_image_dir}", shell=True)
        os.makedirs(preprocessed_image_dir, exist_ok=True)
        preprocess_images(
            f"{workspace_dir}/images", f"{workspace_dir}/preprocessed_images"
        )
    # ---------------------------------------------------------------------------------------------#
    # TODO: CREATE DIFFERENT DATABASES FOR DIFFERENT METHODS
    # 2.A RUN DISK GET THE DATABASE FILE
    disk_handler(workspace_dir)

    # 2.B RUN SIFT GET THE DATABASE FILE
    sift_handler(workspace_dir)

    # 2.c RUN SuperPoint GET THE DATABASE FILE
    superpoint_handler(workspace_dir)

    # 3. DATABASE MERGING
    database_merger_handler(workspace_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline for 3D reconstruction")
    parser.add_argument(
        "--workspace", default="workspace", help="The path to the workspace"
    )

    args = parser.parse_args()
    workspace_dir = args.workspace

    main(workspace_dir=workspace_dir)
