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

# from .colmap.colmap.database import *


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


def lighglue_matcher_handler(workspace_dir: str, method: str) -> bool:
    return


def disk_handler(workspace_dir: str) -> bool:

    img_dir = f"{workspace_dir}/images"

    # disk dir
    disk_dir = f"{workspace_dir}/disk"
    h5_artifacts_destination = f"{disk_dir}/h5"
    database_path = f"{disk_dir}/database.db"

    matching_commands = f"""
    python detect.py {h5_artifacts_destination} {img_dir}
    python match.py --rt 0.95 --save-threshold 100 {h5_artifacts_destination}
    python colmap/h5_to_db.py --database-path {database_path} {h5_artifacts_destination} {img_dir}

    # colmap exhaustive_matcher --database_path {database_path} --SiftMatching.use_gpu 0
    
    """

    output = subprocess.run(matching_commands, shell=True, check=True)

    # # matching the descriptors
    # flann_matcher_handler(workspace_dir, "disk")

    # # TODO: 3D RECONSTRUCTION AND DENSE RECONSTRUCTION

    # reconstruction_handler(workspace_dir, "disk")
    return True


def colmap_sift_handler(workspace_dir: str) -> bool:

    img_dir = f"{workspace_dir}/images"
    colmap_dir = f"{workspace_dir}/colmap"
    database_path = f"{colmap_dir}/database.db"

    matching_commands = f"""
    colmap feature_extractor --ImageReader.single_camera=1 \
        --ImageReader.camera_model=SIMPLE_PINHOLE \
        --database_path {database_path} \
        --image_path {img_dir}\
        --SiftExtraction.use_gpu=1
            
    
    # colmap exhaustive_matcher \
    #     --database_path {database_path} \
    #     --SiftMatching.use_gpu=1
    """
    output = subprocess.run(matching_commands, shell=True, check=True)
    # # matching the descriptors
    # flann_matcher_handler(workspace_dir, "colmap_sift")
    # # TODO: 3D RECONSTRUCTION AND DENSE RECONSTRUCTION
    # reconstruction_handler(workspace_dir, "colmap_sift")

    return True


def keypoint_merge_handler(workspace_dir: str) -> bool:

    # TODO: GET KEYPOINTS FROM COLMAP DATABASE
    # TODO: GET KEYPOINTS FROM DISK DATABASE

    # TODO: GET MATCHES FROM COLMAP DATABASE
    # TODO: GET MATCHES FROM DISK DATABASE

    return True


def main(workspace_dir: str, matcher: str):

    # 1. RUN DISK GET THE DATABASE FILE
    disk_handler(workspace_dir)

    # 2. RUN THE COLMAP CMD TO GET KEKYPOINTS AND DESCRIPTORS
    colmap_sift_handler(workspace_dir)

    pass


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Pipeline for 3D reconstruction")
    parser.add_argument(
        "--workspace", default="workspace", help="The path to the workspace"
    )
    parser.add_argument(
        "--matcher",
        help="The matcher to use for matching the keypoints",
    )

    args = parser.parse_args()
    workspace_dir = args.workspace
    matcher = args.matcher

    main(workspace_dir=workspace_dir, matcher=matcher)
