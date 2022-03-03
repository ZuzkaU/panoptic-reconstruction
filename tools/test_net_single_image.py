import argparse
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from typing import Dict, Any

from lib import modeling

import lib.data.transforms2d as t2d
from lib.config import config
from lib.utils.intrinsics import adjust_intrinsic
from lib.structures import DepthMap

import lib.visualize as vis
from lib.visualize.image import write_detection_image, write_depth


def main(opts):
    configure_inference(opts)

    device = torch.device("cuda:0")

    # Define model and load checkpoint.
    print("Load model...")
    model = modeling.PanopticReconstruction()
    model.load_state_dict(torch.load(opts.model)["model"])  # load model checkpoint
    model = model.to(device)  # move to gpu
    model.switch_test()

    # Define image transformation.
    color_image_size = (320, 240)
    depth_image_size = (160, 120)

    imagenet_stats = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    image_transforms = t2d.Compose([
        t2d.Resize(color_image_size),
        t2d.ToTensor(),
        t2d.Normalize(imagenet_stats[0], imagenet_stats[1]),  # use imagenet stats to normalize image
    ])

    # Open and prepare input image.
    print("Load input image...")
    input_image = Image.open(opts.input, formats=["PNG"])
    input_image = image_transforms(input_image)
    input_image = input_image.unsqueeze(0).to(device)

    # Prepare intrinsic matrix.
    front3d_intrinsic = np.array(config.MODEL.PROJECTION.INTRINSIC)
    front3d_intrinsic = adjust_intrinsic(front3d_intrinsic, color_image_size, depth_image_size)
    front3d_intrinsic = torch.from_numpy(front3d_intrinsic).to(device).float()

    # Prepare frustum mask.
    front3d_frustum_mask = np.load(str("data/frustum_mask.npz"))["mask"]
    front3d_frustum_mask = torch.from_numpy(front3d_frustum_mask).bool().to(device).unsqueeze(0).unsqueeze(0)

    print("Perform panoptic 3D scene reconstruction...")
    with torch.no_grad():
        results = model.inference(input_image, front3d_intrinsic, front3d_frustum_mask)

    print(f"Visualize results, save them at {config.OUTPUT_DIR}")
    visualize_results(results, config.OUTPUT_DIR)


def configure_inference(opts):
    # load config
    config.OUTPUT_DIR = opts.output
    config.merge_from_file(opts.config_file)
    config.merge_from_list(opts.opts)
    # inference settings
    config.MODEL.FRUSTUM3D.IS_LEVEL_64 = False
    config.MODEL.FRUSTUM3D.IS_LEVEL_128 = False
    config.MODEL.FRUSTUM3D.IS_LEVEL_256 = False
    config.MODEL.FRUSTUM3D.FIX = True


def visualize_results(results: Dict[str, Any], output_path: os.PathLike) -> None:
    device = results["depth"].device
    output_path = Path(output_path)
    output_path.mkdir(exist_ok=True, parents=True)

    # Visualize depth prediction
    depth_map = DepthMap(results["depth"].squeeze(), results["intrinsic"])
    depth_map.to_pointcloud(output_path / "/depth_prediction.ply")
    write_depth(depth_map, output_path / "depth_map.png")

    # Visualize 2D detections
    write_detection_image(results["input"], results["instance"], output_path / "detection.png")

    # Visualize projection
    vis.write_pointcloud(results["projection"].C[:, 1:], None, output_path / "projection.ply")

    # Visualize 3D outputs
    dense_dimensions = torch.Size([1, 1] + config.MODEL.FRUSTUM3D.GRID_DIMENSIONS)
    min_coordinates = torch.IntTensor([0, 0, 0]).to(device)
    truncation = config.MODEL.FRUSTUM3D.TRUNCATION
    iso_value = config.MODEL.FRUSTUM3D.ISO_VALUE

    geometry = results["frustum"]["geometry"]
    surface, _, _ = geometry.dense(dense_dimensions, min_coordinates, default_value=truncation)
    semantics, _, _ = results["panoptic"]["panoptic_semantics"].dense(dense_dimensions, min_coordinates)
    instances, _, _ = results["panoptic"]["panoptic_instances"].dense(dense_dimensions, min_coordinates)

    # Visualize sparse outputs
    vis.write_pointcloud(geometry.C[:, 1:], None, output_path / "sparse_coordinates.ply")

    points = (surface < iso_value).squeeze().nonzero()
    vis.write_pointcloud(points, None, output_path / "points_geometry.ply")
    vis.write_semantic_pointcloud(points, None, output_path / "surface_pcd.ply")
    vis.write_semantic_pointcloud(points, None, output_path / "surface_pcd.ply")

    vis.write_distance_field(surface.squeeze(), None, output_path / "mesh_geometry.ply")
    vis.write_distance_field(surface.squeeze(), semantics.squeeze(), output_path / "mesh_semantics.ply")
    vis.write_distance_field(surface.squeeze(), instances.squeeze(), output_path / "mesh_instances.ply")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", type=str, required=True, default="data/front3d-sample/rgb_0007.png")
    parser.add_argument("--output", "-o", type=str, required=True, default="output/sample_0007/")
    parser.add_argument("--config-file", "-c", type=str, default="configs/front3d_sample.yaml")
    parser.add_argument("--model", "-m", type=str, default="data/panoptic_front3d_v2.pth")
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER)

    args = parser.parse_args()

    main(args)
