# SPDX-FileCopyrightText: © 2023 Project's authors 
# SPDX-License-Identifier: MIT

import argparse
import json
import LineMesh
from utils import (get_trafo_from_matches, 
                   load_mesh, 
                   capture_image,
                   QUESTION_ID_TRAFOS,
                   QUESTION_ID_TRANSLATIONS,
                   SCENE_ID_SCALES)
import numpy as np
import open3d as o3d
import os.path as osp
import copy

parser=argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# Couches are ideal for visualization. Examples of questions with couches: val-scene0019-42, val-scene0025-22, val-scene0050-20, val-scene0064-51, val-scene0081-51
parser.add_argument("-q", "--question_id", metavar="STRING", help="The ID of the question for the visualization", type=str, default="val-scene0019-42")
parser.add_argument("--ground_truth_color", metavar="STRING", help="The HEX color of the bounding box of the ground truth", type=lambda color_hex: [int(color_hex[index:index+2], 16)/255 for index in (1, 3, 5)], default="#00ff00")
parser.add_argument("--prediction_color", metavar="STRING", help="The HEX color of the bounding box of the prediction", type=lambda color_hex: [int(color_hex[index:index+2], 16)/255 for index in (1, 3, 5)], default="#ff0000")
parser.add_argument("-r", "--radius", metavar="FLOAT", help="The radius of the cylinders of the bounding box", type=int, default=0.05)
parser.add_argument("--headless", action="store_true", help="If no display avaiable, try this (not tested)")

parser.add_argument("--use_ply", action="store_true", help="Use high-res ply files instead of low-res npy files if exists")
parser.add_argument("--use_v1", action="store_true", help="Use v1 ply files")
parser.add_argument("--ply_path", metavar="STRING", help="Special path to ply files", type=str, default=None)
args=parser.parse_args()

with open("ScanQA_v1.0_val.json","r") as dataset_json:
    dataset = json.load(dataset_json)

# Find the question with the given question_id
for question in dataset:
    if question["question_id"] == args.question_id:
        break

# Obtain the relevant scene_id
scene_id = question["scene_id"]

with open("pred.val.json","r") as predictions_json:
    predictions = json.load(predictions_json)

# Find the prediction for the given question_id
for prediction in predictions:
    if prediction["question_id"] == args.question_id:
        break



# Visualize the ground truth
ground_truth_Vector3dVector = o3d.utility.Vector3dVector(prediction["gt_bbox"])
ground_truth_OrientedBoundingBox = o3d.geometry.OrientedBoundingBox.create_from_points(ground_truth_Vector3dVector)
ground_truth_OrientedBoundingBox.color = args.ground_truth_color
ground_truth_LineSet = o3d.geometry.LineSet.create_from_oriented_bounding_box(ground_truth_OrientedBoundingBox)
ground_truth_LineMesh = LineMesh.LineMesh(ground_truth_LineSet.points, ground_truth_LineSet.lines, ground_truth_LineSet.colors, radius=args.radius)
ground_truth_cylinders = ground_truth_LineMesh.cylinder_segments

# Visualize the prediction
prediction_Vector3dVector = o3d.utility.Vector3dVector(prediction["bbox"])
prediction_OrientedBoundingBox = o3d.geometry.OrientedBoundingBox.create_from_points(prediction_Vector3dVector)
prediction_OrientedBoundingBox.color = args.prediction_color
prediction_LineSet = o3d.geometry.LineSet.create_from_oriented_bounding_box(prediction_OrientedBoundingBox)
prediction_LineMesh = LineMesh.LineMesh(prediction_LineSet.points, prediction_LineSet.lines, prediction_LineSet.colors, radius=args.radius)
prediction_cylinders = prediction_LineMesh.cylinder_segments


use_ply = args.use_ply
if use_ply:
    print("Trying to load the scene's point cloud from a ply file."
          "If it fails, it will fall back to the npy files."
          "It needs a _vert.npy, _aligned_vert.npy and the .ply file.")
    if args.ply_path is not None:
        pc, worked = load_mesh(scene_id, args.ply_path, not args.use_v1)
    else:
        pc, worked = load_mesh(scene_id, use_v2=not args.use_v1)
    if not worked:
        use_ply = False

    if use_ply:
        use_ply = osp.isfile(osp.join("scannet_data", scene_id + "_vert.npy"))
        if not use_ply:
            print("The non-aligned vertices were not found. Falling back to npy files.")

# Obtain the relevant scene_id
scene_id = question["scene_id"]

# List of possible answers
answers = question["answers"]

# As string
question = question["question"]

# load top10 answers of the model
top10s = prediction["answer_top10"]

# Get the trafo
if args.question_id in QUESTION_ID_TRAFOS.keys():
    trafo = QUESTION_ID_TRAFOS[args.question_id]
else:
    trafo = (-2.5*np.pi/9, 0, np.pi/2)

# Get the scale
if scene_id in SCENE_ID_SCALES.keys():
    scale = SCENE_ID_SCALES[scene_id]
else:
    scale = 1.5

# Get the translation
if args.question_id in QUESTION_ID_TRANSLATIONS.keys():
    translation = QUESTION_ID_TRANSLATIONS[args.question_id]
else:
    translation = (0, 0, 0) 

print(f"Scaled by {scale} and rotated by {trafo}")

if not use_ply:
    # For the training/visualization we use the scene's ALIGNED point cloud, stored in a .npy file. 
    scene_xyzrgb = np.load(osp.join("scannet_data", scene_id + "_aligned_vert.npy"))[:, 0:6] 
    scene_xyz = o3d.utility.Vector3dVector(scene_xyzrgb[:, 0:3])
    scene_rgb = o3d.utility.Vector3dVector(scene_xyzrgb[:, 3:6] / 255)
    scene_PointCloud = o3d.geometry.PointCloud(scene_xyz)
    scene_PointCloud.colors = scene_rgb

    # In my first attempt, I obtained the ground truth bounding box from the *_aligned_bbox.npy files.
    # Nevertheless, the alternative below is more elegant (they both give the exact same results).
    # ground_truths_all = np.load("scannet_data/" + scene_id + "_aligned_bbox.npy")
    # object_ids = question["object_ids"]
    # ground_truths_mask = np.isin(ground_truths_all[:, 7], object_ids)
    # ground_truths = ground_truths_all[ground_truths_mask, :]
    # ground_truths_cylinders = []
    # for ground_truth in ground_truths:
    #     ground_truth_OrientedBoundingBox = o3d.geometry.OrientedBoundingBox(ground_truth[0:3].reshape(3,1), np.identity(3), ground_truth[3:6].reshape(3,1))
    #     ground_truth_OrientedBoundingBox.color = args.ground_truth_color
    #     ground_truth_LineSet = o3d.geometry.LineSet.create_from_oriented_bounding_box(ground_truth_OrientedBoundingBox)
    #     ground_truth_LineMesh = LineMesh.LineMesh(ground_truth_LineSet.points, ground_truth_LineSet.lines, ground_truth_LineSet.colors, radius=args.radius)
    #     ground_truth_cylinders = ground_truth_LineMesh.cylinder_segments
    #     ground_truths_cylinders = [*ground_truths_cylinders, *ground_truth_cylinders]
    # o3d.visualization.draw_geometries([scene_PointCloud, *ground_truths_cylinders])

    geoms_gt = [copy.deepcopy(scene_PointCloud), *ground_truth_cylinders]
    geoms_pred = [copy.deepcopy(scene_PointCloud), *prediction_cylinders]

    # We need a source pc for transformations
    source_pc = scene_PointCloud
else:
    num = 500 # We only need a few points to get accurate matches (every 500 points still gives more than 10 points which is enough) 
    scene_xyz_al = np.load(osp.join("scannet_data", scene_id + "_aligned_vert.npy"))[:, 0:3][::num]
    scene_xyz_na = np.load(osp.join("scannet_data", scene_id + "_vert.npy"))[:, 0:3][::num] 

    # Use matches to generate the trafo matrix
    T = get_trafo_from_matches(scene_xyz_na, scene_xyz_al)

    # transform pointcloud into aligned form
    pc = pc.transform(T)

    # Register the geometry objects
    geoms_gt = [copy.deepcopy(pc), *ground_truth_cylinders]
    geoms_pred = [copy.deepcopy(pc), *prediction_cylinders]

    # We need a source pc for transformations
    source_pc = pc

# Load visualizer
vis = o3d.visualization.VisualizerWithKeyCallback()
vis.create_window(width=1280, height=720, visible=True if not args.headless else False)
vis.get_render_option().background_color = np.asarray([256,256,256])

# Transfrom and add the geometry objects to the visualizer and take a screenshot
for geom in geoms_gt:
    R = source_pc.get_rotation_matrix_from_xyz(trafo)
    geom = geom.rotate(R, center=source_pc.get_center())
    geom = geom.translate(translation)
    vis.add_geometry(geom)
    if scale is not None:
        geom = geom.scale(scale, center=source_pc.get_center())
    vis.poll_events()
    vis.update_geometry(geom)

for _ in range(100):
    vis.poll_events()
    vis.update_renderer()

capture_image(vis, f"{scene_id}_{args.question_id}_ground_truth.png")

for geom in geoms_gt:
    vis.remove_geometry(geom)

# Transfrom and add the geometry objects to the visualizer and take a screenshot
for geom in geoms_pred:
    R = source_pc.get_rotation_matrix_from_xyz(trafo)
    geom = geom.rotate(R, center=source_pc.get_center())
    geom = geom.translate(translation)
    vis.add_geometry(geom)
    if scale is not None:
        geom = geom.scale(scale, center=source_pc.get_center())
    vis.poll_events()
    vis.update_geometry(geom)

for _ in range(100):
    vis.poll_events()
    vis.update_renderer()

capture_image(vis, f"{scene_id}_{args.question_id}_pred.png")

vis.close()

# Save the question, question_id, scene_id, answers, ground truth and prediction in a .txt file
with open(f"{scene_id}_{args.question_id}.txt", "w+") as file:
    file.write(f"Question: {question}\n")
    file.write(f"Question ID: {args.question_id}\n")
    file.write(f"Scene ID: {scene_id}\n")
    file.write(f"True Answers: {answers}\n")
    file.write(f"Predicted top10: {top10s}\n")
