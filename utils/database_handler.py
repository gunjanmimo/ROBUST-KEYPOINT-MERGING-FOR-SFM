import sys
import sqlite3
import numpy as np
from colmap.colmap import database

MAX_IMAGE_ID = 2**31 - 1


def blob_to_array(blob, dtype, shape=(-1,)):
    if blob is None:
        return []

    return np.fromstring(blob, dtype=dtype).reshape(*shape)


def pair_id_to_image_ids(pair_id):
    image_id2 = pair_id % MAX_IMAGE_ID
    image_id1 = (pair_id - image_id2) / MAX_IMAGE_ID
    return image_id1, image_id2


def get_conn(database_path: str):
    conn = sqlite3.connect(database_path)
    cursor = conn.cursor()
    return conn, cursor


def get_base_keypoints(cursor, method: str):
    # preprocess keypoints
    if method == "disk":
        keypoints = dict(
            (image_id, blob_to_array(data, np.float32, (-1, 2)))
            for image_id, data in cursor.execute("SELECT image_id, data FROM keypoints")
        )
    elif method == "colmap":
        keypoints = dict(
            (image_id, blob_to_array(data, np.float32, (-1, 6))[:, :2])
            for image_id, data in cursor.execute("SELECT image_id, data FROM keypoints")
        )
    return keypoints


def get_base_keypoints_by_image_id(cursor, method: str, image_id: int):
    # preprocess keypoints
    if method == "disk":
        keypoints = dict(
            (image_id, blob_to_array(data, np.float32, (-1, 2)))
            for image_id, data in cursor.execute(
                "SELECT image_id, data FROM keypoints WHERE image_id = ?", (image_id,)
            )
        )
    elif method == "colmap":
        keypoints = dict(
            (image_id, blob_to_array(data, np.float32, (-1, 6))[:, :2])
            for image_id, data in cursor.execute(
                "SELECT image_id, data FROM keypoints WHERE image_id = ?", (image_id,)
            )
        )
    return keypoints


def get_base_matches(cursor):
    matches = dict(
        (pair_id, blob_to_array(data, np.uint32, (-1, 2)))
        for pair_id, data in cursor.execute("SELECT pair_id, data FROM matches")
    )
    return matches


def copy_colmap_database(colmap_db_path: str, merged_db_path: str):
    # Connect to the source COLMAP database
    colmap_conn = sqlite3.connect(colmap_db_path)
    colmap_cursor = colmap_conn.cursor()

    # Connect to the target (merged) COLMAP database
    merged_conn = sqlite3.connect(merged_db_path)
    merged_cursor = merged_conn.cursor()

    # Copy data from the source to the target database
    # For cameras table
    colmap_cursor.execute("SELECT * FROM cameras")
    cameras = colmap_cursor.fetchall()
    merged_cursor.executemany(
        "INSERT OR IGNORE INTO cameras VALUES (?, ?, ?, ?, ?, ?)", cameras
    )

    # For images table
    colmap_cursor.execute("SELECT * FROM images")
    images = colmap_cursor.fetchall()
    merged_cursor.executemany(
        "INSERT OR IGNORE INTO images VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", images
    )

    # Commit the changes and close the connections
    merged_conn.commit()
    colmap_conn.close()
    merged_conn.close()


def merge_keypoint(
    colmap_database_path: str,
    disk_database_path: str,
    merged_database_path: str,
):

    # connection
    colmap_conn, colmap_cursor = get_conn(colmap_database_path)
    disk_conn, disk_cursor = get_conn(disk_database_path)

    merged_db = database.COLMAPDatabase.connect(merged_database_path)

    # colmap keypoints
    colmap_keypoints = get_base_keypoints(colmap_cursor, "disk")
    # disk keypoints
    disk_keypoints = get_base_keypoints(disk_cursor, "disk")

    # merge the keypoints
    merged_keypoints = {}
    for image_id, keypoints in colmap_keypoints.items():

        # first colmap keypoints are added
        # then disk keypoints are added
        # index of colmap keypoints are not changed
        # index of disk keypoints are changed
        # current disk keypoint index are = previous colmap keypoint index + previous colmap keypoint count

        if image_id in disk_keypoints:
            merged_keypoints[image_id] = np.concatenate(
                (keypoints, disk_keypoints[image_id]), axis=0
            )
        else:
            merged_keypoints[image_id] = keypoints

    # insert the merged keypoints
    for image_id, keypoints in merged_keypoints.items():
        merged_db.add_keypoints(image_id, keypoints)

    # commit the changes
    merged_db.commit()

    # close the connections
    colmap_conn.close()
    disk_conn.close()
    merged_db.close()


def merge_matches(
    colmap_database_path: str,
    disk_database_path: str,
    merged_database_path: str,
):

    # connection
    colmap_conn, colmap_cursor = get_conn(colmap_database_path)
    disk_conn, disk_cursor = get_conn(disk_database_path)
    merged_db = database.COLMAPDatabase.connect(merged_database_path)

    # matches
    colmap_matches = get_base_matches(colmap_cursor)
    disk_matches = get_base_matches(disk_cursor)

    # merge the matches
    merged_matches = {}
    for pair_id, matches in colmap_matches.items():
        image_id1, image_id2 = pair_id_to_image_ids(pair_id)
        # if pair_id in disk_matches:
        total_colmap_keypoints_img1 = get_base_keypoints_by_image_id(
            colmap_cursor, "disk", image_id1
        )[image_id1].shape[0]
        total_colmap_keypoints_img2 = get_base_keypoints_by_image_id(
            colmap_cursor, "disk", image_id2
        )[image_id2].shape[0]

        # print("pair_id", pair_id)
        # print("image_id1", image_id1)
        # print("image_id2", image_id2)
        print("total_colmap_keypoints_img1", total_colmap_keypoints_img1)
        print("total_colmap_keypoints_img2", total_colmap_keypoints_img2)

        print(f"total colmap_match: {len(matches)}")
        print(f"total disk_match: {len(disk_matches[pair_id])}")

        disk_match = disk_matches[pair_id]
        if len(disk_match) > 0:
            disk_match[:, 0] += total_colmap_keypoints_img1
            disk_match[:, 1] += total_colmap_keypoints_img2

            # merge the matches
            merged_matches[pair_id] = np.concatenate((matches, disk_match), axis=0)
        else:
            merged_matches[pair_id] = matches

        print(f"total merged_match: {len(merged_matches[pair_id])}")

        print("_" * 50)
        # else:
        #     merged_matches[pair_id] = matches

    # insert the merged matches
    for pair_id, matches in merged_matches.items():
        if len(matches) > 0:
            merged_db.add_matches(
                image_id1=pair_id_to_image_ids(pair_id)[0],
                image_id2=pair_id_to_image_ids(pair_id)[1],
                matches=matches,
            )
            merged_db.add_two_view_geometry(
                image_id1=pair_id_to_image_ids(pair_id)[0],
                image_id2=pair_id_to_image_ids(pair_id)[1],
                matches=matches,
            )
    # commit the changes
    merged_db.commit()
    # close the connections
    colmap_conn.close()
    disk_conn.close()
    merged_db.close()
