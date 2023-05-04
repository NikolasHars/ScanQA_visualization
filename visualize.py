# SPDX-FileCopyrightText: © 2023 Georgios Vlassis <gvlassis@mailbox.org> 
# SPDX-License-Identifier: MIT

import argparse
import json
import numpy
import open3d
import LineMesh

parser=argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# Couches are ideal for visualization. Examples of questions with couches: val-scene0019-42, val-scene0025-22, val-scene0050-20, val-scene0064-51, val-scene0081-51
parser.add_argument("-q", "--question_id", metavar="STRING", help="The ID of the question for the visualization", type=str, default="val-scene0019-42")
parser.add_argument("--ground_truth_color", metavar="STRING", help="The HEX color of the bounding box of the ground truth", type=lambda color_hex: [int(color_hex[index:index+2], 16)/255 for index in (1, 3, 5)], default="#00ff00")
parser.add_argument("--prediction_color", metavar="STRING", help="The HEX color of the bounding box of the prediction", type=lambda color_hex: [int(color_hex[index:index+2], 16)/255 for index in (1, 3, 5)], default="#ff0000")
parser.add_argument("-r", "--radius", metavar="FLOAT", help="The radius of the cylinders of the bounding box", type=int, default=0.05)
args=parser.parse_args()

with open("ScanQA_v1.0_val.json","r") as dataset_json:
    dataset = json.load(dataset_json)

# Find the question with the given question_id
for question in dataset:
    if question["question_id"] == args.question_id:
        break

# Obtain the relevant scene_id
scene_id = question["scene_id"]

# For the training/visualization we use the scene's ALIGNED point cloud, stored in a .npy file. 
scene_xyzrgb = numpy.load("scannet_data/" + scene_id + "_aligned_vert.npy")[:, 0:6] 
scene_xyz = open3d.utility.Vector3dVector(scene_xyzrgb[:, 0:3])
scene_rgb = open3d.utility.Vector3dVector(scene_xyzrgb[:, 3:6] / 255)
scene_PointCloud = open3d.geometry.PointCloud(scene_xyz)
scene_PointCloud.colors = scene_rgb

# In my first attempt, I obtained the ground truth bounding box from the *_aligned_bbox.npy files.
# Nevertheless, the alternative below is more elegant (they both give the exact same results).
# ground_truths_all = numpy.load("scannet_data/" + scene_id + "_aligned_bbox.npy")
# object_ids = question["object_ids"]
# ground_truths_mask = numpy.isin(ground_truths_all[:, 7], object_ids)
# ground_truths = ground_truths_all[ground_truths_mask, :]
# ground_truths_cylinders = []
# for ground_truth in ground_truths:
#     ground_truth_OrientedBoundingBox = open3d.geometry.OrientedBoundingBox(ground_truth[0:3].reshape(3,1), numpy.identity(3), ground_truth[3:6].reshape(3,1))
#     ground_truth_OrientedBoundingBox.color = args.ground_truth_color
#     ground_truth_LineSet = open3d.geometry.LineSet.create_from_oriented_bounding_box(ground_truth_OrientedBoundingBox)
#     ground_truth_LineMesh = LineMesh.LineMesh(ground_truth_LineSet.points, ground_truth_LineSet.lines, ground_truth_LineSet.colors, radius=args.radius)
#     ground_truth_cylinders = ground_truth_LineMesh.cylinder_segments
#     ground_truths_cylinders = [*ground_truths_cylinders, *ground_truth_cylinders]
# open3d.visualization.draw_geometries([scene_PointCloud, *ground_truths_cylinders])

with open("pred.val.json","r") as predictions_json:
    predictions = json.load(predictions_json)

# Find the prediction for the given question_id
for prediction in predictions:
    if prediction["question_id"] == args.question_id:
        break

# Visualize the ground truth
ground_truth_Vector3dVector = open3d.utility.Vector3dVector(prediction["gt_bbox"])
ground_truth_OrientedBoundingBox = open3d.geometry.OrientedBoundingBox.create_from_points(ground_truth_Vector3dVector)
ground_truth_OrientedBoundingBox.color = args.ground_truth_color
ground_truth_LineSet = open3d.geometry.LineSet.create_from_oriented_bounding_box(ground_truth_OrientedBoundingBox)
ground_truth_LineMesh = LineMesh.LineMesh(ground_truth_LineSet.points, ground_truth_LineSet.lines, ground_truth_LineSet.colors, radius=args.radius)
ground_truth_cylinders = ground_truth_LineMesh.cylinder_segments
open3d.visualization.draw_geometries([scene_PointCloud, *ground_truth_cylinders])

# Visualize the prediction
prediction_Vector3dVector = open3d.utility.Vector3dVector(prediction["bbox"])
prediction_OrientedBoundingBox = open3d.geometry.OrientedBoundingBox.create_from_points(prediction_Vector3dVector)
prediction_OrientedBoundingBox.color = args.prediction_color
prediction_LineSet = open3d.geometry.LineSet.create_from_oriented_bounding_box(prediction_OrientedBoundingBox)
prediction_LineMesh = LineMesh.LineMesh(prediction_LineSet.points, prediction_LineSet.lines, prediction_LineSet.colors, radius=args.radius)
prediction_cylinders = prediction_LineMesh.cylinder_segments
open3d.visualization.draw_geometries([scene_PointCloud, *prediction_cylinders])