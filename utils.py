import numpy as np
import numpy as np
import open3d as o3d
from PIL import Image
import os

from typing import List, Tuple, Dict, Optional, Union

# Update these dictionary with the transformations of the point clouds of the questions
QUESTION_ID_TRAFOS = {
    "val-scene0019-42": (-2.5*np.pi/9, 0, 1+np.pi/2),
    "val-scene0050-20": (-2.5*np.pi/9, 0, 1.1-np.pi/2),
    "val-scene0019-59": (-2.5*np.pi/9, 0, 1+np.pi/2),
    "val-scene0025-5": (-2.5*np.pi/9, 0, np.pi - np.pi/9),
    "val-scene0025-22": (-2.5*np.pi/9, 0, - np.pi/9),
}

QUESTION_ID_TRANSLATIONS = {
    "val-scene0025-22": (0, 1.5, 0),
    "val-scene0019-42": (0, 1, 0),
    "val-scene0050-20": (0, 1, 0),
}

SCENE_ID_SCALES = {
    "scene0019_00": 1.75,
    "scene0025_00": 1.5,
    "scene0050_00": 1.75,
}

def load_mesh(scene: str,
              scene_path: Optional[str] = "scannet_data",
              use_v2: Optional[bool] = True
              ) -> Tuple[o3d.geometry.TriangleMesh, bool]:
    """loads pointcloud from the scene_vh_clean_2.ply file

    Args:
        scene (str): scene_id e.g. scene0019_00
        scene_path (Optional[str], optional): speficy where the ply file is stored. Defaults to "scannnet_data".
        use_v2 (bool, optional): use the _vh_clean_2.ply or the _vh_clean.ply file. Defaults to True.

    Returns:
        Tuple[open3d.geometry.TriangleMesh, bool]: TriangleMesh object containing the scene pointcloud and bool to indicate if the file was found
    """
    suffix = "_vh_clean_2.ply" if use_v2 else "_vh_clean.ply"
    pcd_path = os.path.join(scene_path, scene + suffix)
    if not os.path.isfile(pcd_path):
        print("Couldn't find file at path: ", pcd_path)
        print("Fallback to aligned npy file")
        return None, False
    source = o3d.io.read_triangle_mesh(pcd_path, enable_post_processing=False, print_progress=False)
    return source, True

def capture_image(vis: o3d.visualization.Visualizer, image_path: str) -> None:
    """vaptures current rendered image from the visaulizer and saves it to the specified path

    Args:
        vis (open3d.visualization.Visualizer): Visualizer object
        image_path (str): path to save the image to

    Returns:
        None
    """
    image = vis.capture_screen_float_buffer()
    image = np.asarray(image) * 255
    pil_img = Image.fromarray(image.astype(np.uint8))
    pil_img.save(image_path)


def get_trafo_from_matches(P: np.ndarray, Q: np.ndarray) -> np.ndarray:
    """Computes the rigid transformation between two !!matching!! point clouds P and Q. The resulting transformation is from P to Q

    Args:
        P (np.ndarray): (N, 3) Source point cloud
        Q (np.ndarray): (N, 3) Target point cloud

    Returns:
        np.ndarray: (4,4) transformation matrix from P to Q (homogeneous coordinates)
    """
    # Compute the centroids
    C1 = np.mean(P, axis=0)
    C2 = np.mean(Q, axis=0)

    # Compute the centered points
    Pc = P - C1
    Qc = Q - C2

    # Compute the matrix H
    H = np.dot(Pc.T, Qc)

    # Compute the Singular Value Decomposition of H
    U, S, Vt = np.linalg.svd(H)

    # Compute the rotation matrix R
    R = np.dot(Vt.T, U.T)

    # Compute the translation vector T
    T = C2 - np.dot(R, C1)

    # Compute the transformation matrix
    T_final = np.eye(4)
    T_final[:3, :3] = R
    T_final[:3, 3] = T

    print(T_final)
    return T_final
