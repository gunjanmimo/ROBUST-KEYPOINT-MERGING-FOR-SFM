"""

TODO:
    1. RUN DISK GET THE DATABASE FILE 
    2. RUN THE COLMAP CMD TO GET KEKYPOINTS AND DESCRIPTORS
    3. USE LIGHTGLUE TO MATCH THE DESCRIPTORS OF SIFT
    4. USE COLMAP TO RECONSTRUCT THE 3D MODEL
    
"""

"""
EXPERIMENT DESIGN 

1. COLMAP GPU SIFT + DISK ------- FLANN MATCHING
2. COLMAP GPU SIFT + DISK ------- LIGHTGLUE MATCHING


FOLDER STRUCTURE 

.
└── workspace/
    ├── images
    ├── disk /
    │   ├── h5/
    │   │   ├── keypoints.h5 
    │   │   └── matches.h5
    │   └── database.db
    └── colmap/
        ├── database.db
        ├── sparse/
        │   ├── sparse.ply
        │   └── scene/
        │       └── 0  
        └── dense /
            ├── fuesed.ply
            └── sparse/
                ├── 0
                ├── 1
                └── 2

"""

import os
import subprocess
import argparse
import sqlite3
from PIL import Image
import cv2
from tqdm import tqdm


# local imports
from colmap.colmap import database
from utils import database_handler


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


def reconstruction_handler(workspace_dir: str, method: str) -> bool:
    database_path = f"{workspace_dir}/{method}/database.db"
    image_dir = f"{workspace_dir}/images"
    spare_dir = f"{workspace_dir}/{method}/sparse"
    dense_dir = f"{workspace_dir}/{method}/dense"

    os.makedirs(spare_dir, exist_ok=True)
    os.makedirs(dense_dir, exist_ok=True)

    sparse_ply_path = f"{spare_dir}/sparse.ply"
    dense_ply_path = f"{dense_dir}/fused.ply"

    reconstruction_commands = f"""
    
    # TODO: SPARSE RECONSTRUCTION
    colmap mapper \
            --database_path {database_path} \
            --image_path {image_dir} \
            --output_path {spare_dir}
            
            
    colmap model_converter \
            --input_path {spare_dir}/0 \
            --output_path {sparse_ply_path} \
            --output_type PLY
    
    # TODO: DENSE RECONSTRUCTION
    
    
    colmap image_undistorter \
            --image_path {image_dir} \
            --input_path {spare_dir}/0 \
            --output_path {dense_dir} \
            --output_type COLMAP \
            --max_image_size 4000
            
        colmap patch_match_stereo \
            --workspace_path {dense_dir} \
            --workspace_format COLMAP \
            --PatchMatchStereo.geom_consistency true
        
        colmap stereo_fusion \
            --workspace_path {dense_dir} \
            --workspace_format COLMAP \
            --input_type geometric \
            --output_path {dense_dir}/fused.ply
        
        colmap poisson_mesher \
            --input_path {dense_dir}/fused.ply \
            --output_path {dense_dir}/meshed-poisson.ply
        
        colmap delaunay_mesher \
            --input_path {dense_dir}/fused.ply \
            --output_path {dense_dir}/meshed-delaunay.ply
    
    
    
    """
    pass


def flann_matcher_handler(workspace_dir: str, method: str) -> bool:
    workspace_dir = f"{workspace_dir}/{method}"
    database_path = f"{workspace_dir}/database.db"

    matching_commands = f"""
    colmap exhaustive_matcher \
        --database_path {database_path} \
        --SiftMatching.use_gpu=1
    
    """

    output = subprocess.run(matching_commands, shell=True, check=True)

    return True


def disk_handler(workspace_dir: str) -> bool:

    img_dir = f"{workspace_dir}/preprocessed_images"

    # disk dir
    disk_dir = f"{workspace_dir}/disk"
    os.makedirs(disk_dir, exist_ok=True)
    h5_artifacts_destination = f"{disk_dir}/h5"
    database_path = f"{disk_dir}/database.db"
    # remove existing database file
    if os.path.exists(database_path):
        os.remove(database_path)

    matching_commands = f"""
    python detect.py --height 1024 --width 1024 {h5_artifacts_destination} {img_dir}
    python match.py --rt 0.95 --save-threshold 100 {h5_artifacts_destination}
    python colmap/h5_to_db.py --database-path {database_path} {h5_artifacts_destination} {img_dir} --camera-model='simple-pinhole' --single-camera
    """
    output = subprocess.run(matching_commands, shell=True, check=True)
    return True


def colmap_sift_handler(workspace_dir: str) -> bool:

    img_dir = f"{workspace_dir}/preprocessed_images"
    colmap_dir = f"{workspace_dir}/colmap"
    os.makedirs(colmap_dir, exist_ok=True)
    database_path = f"{colmap_dir}/database.db"

    matching_commands = f"""
    colmap feature_extractor --ImageReader.single_camera=1 \
        --ImageReader.camera_model=SIMPLE_PINHOLE \
        --database_path {database_path} \
        --image_path {img_dir}\
        --SiftExtraction.use_gpu=1

    """
    output = subprocess.run(matching_commands, shell=True, check=True)
    return True


def keypoint_merge_handler(workspace_dir: str) -> bool:

    # TODO: GET KEYPOINTS FROM COLMAP DATABASE
    # TODO: GET KEYPOINTS FROM DISK DATABASE

    # TODO: GET MATCHES FROM COLMAP DATABASE
    # TODO: GET MATCHES FROM DISK DATABASE

    return True


def main(workspace_dir: str):

    # # 1. PREPROCESS THE IMAGES
    # preprocessed_image_dir = f"{workspace_dir}/preprocessed_images"
    # # delete if the directory exists
    # if os.path.exists(preprocessed_image_dir):
    #     subprocess.run(f"rm -rf {preprocessed_image_dir}", shell=True)

    # os.makedirs(preprocessed_image_dir, exist_ok=True)
    # preprocess_images(f"{workspace_dir}/images", f"{workspace_dir}/preprocessed_images")
    # ---------------------------------------------------------------------------------------------#
    # 2. RUN DISK GET THE DATABASE FILE
    disk_handler(workspace_dir)

    # 3. RUN THE COLMAP CMD TO GET KEYPOINTS AND DESCRIPTORS
    colmap_sift_handler(workspace_dir)
    # ---------------------------------------------------------------------------------------------#
    # 4. MATCHING KEYPOINTS
    flann_matcher_handler(workspace_dir, "colmap")
    flann_matcher_handler(workspace_dir, "disk")

    # ---------------------------------------------------------------------------------------------#

    # 5. CREATE A DATABASE FOR THE MERGED KEYPOINTS
    print("::: DATABASE MERGING :::")
    merged_database_path = f"{workspace_dir}/merged_database.db"
    # delete database file if it exists
    if os.path.exists(merged_database_path):
        os.remove(merged_database_path)

    db = database.COLMAPDatabase.connect(merged_database_path)
    db.create_tables()

    # copy camera and image table from colmap database
    """
    IT IS IMPORTANT TO COPY THE CAMERA AND IMAGE TABLES FROM THE COLMAP DATABASE TO THE MERGED DATABASE
    """
    database_handler.copy_colmap_database(
        f"{workspace_dir}/colmap/database.db", merged_database_path
    )
    # merge the keypoints
    database_handler.merge_keypoint(
        f"{workspace_dir}/colmap/database.db",
        f"{workspace_dir}/disk/database.db",
        merged_database_path,
    )
    # merge the matches
    database_handler.merge_matches(
        f"{workspace_dir}/colmap/database.db",
        f"{workspace_dir}/disk/database.db",
        merged_database_path,
    )

    # close the database
    db.close()
    # ---------------------------------------------------------------------------------------------#

    pass


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Pipeline for 3D reconstruction")
    parser.add_argument(
        "--workspace", default="workspace", help="The path to the workspace"
    )

    args = parser.parse_args()
    workspace_dir = args.workspace

    main(workspace_dir=workspace_dir)
