""" My first Gaussian render
adapted from https://github.com/nerfstudio-project/gsplat/blob/main/examples/simple_viewer.py

"""

import argparse
import math
import os
import hydra
import time
from typing import Tuple

import imageio
import nerfview
import numpy as np
import tqdm
import viser

from omegaconf import DictConfig
from open3d import *
# from gsplat._helper import load_test_data
# from gsplat.distributed import cli
# from gsplat.rendering import rasterization


@hydra.main(version_base=None, config_path="./config/", config_name="config")
def main(cfg: DictConfig):
    
    point_cloud = io.read_point_cloud(cfg.paths.ply_path) # Read point cloud
    point_cloud_array = np.asarray(point_cloud.points)
    point_cloud_colors = np.asarray(point_cloud.colors)
    point_cloud_array_homogeneous = np.hstack((point_cloud_array, np.ones((point_cloud_array.shape[0], 1))))
    
    print("here only once right?")
        
    def viewer_render_fn(camera_state: nerfview.CameraState, img_wh: Tuple[int, int]):
        width, height = img_wh
        c2w = camera_state.c2w
        viewmat = np.linalg.inv(c2w)
        K = camera_state.get_K(img_wh)
        
        print("c2w", viewmat[:3, :4])
        print("K", K)
        
        # fx = fy = 0.5 * width  
        # cx = width / 2
        # cy = height / 2
        
        # K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        
        P = np.dot(K, viewmat[:3, :4])
        
        # points
        point_2d = np.dot(P, point_cloud_array_homogeneous.T).T
        point_2d /= point_2d[:, 2].reshape(-1, 1)  # Normalize homogeneous coordinates
        
        # Get x and y coordinates
        x = point_2d[:, 0]
        y = point_2d[:, 1]

        # Create mask for points inside the image boundaries
        mask = (x >= 0) & (x < width) & (y >= 0) & (y < height)

        # Apply mask to x, y, and colors
        x = x[mask]
        y = y[mask]
        colors = point_cloud_colors[mask]

        # Convert x and y to integer pixel indices
        x_int = x.astype(int)
        y_int = y.astype(int)

        # Initialize the image
        image = np.ones((height, width, 3), dtype=np.uint8) * 255

        # Render the points onto the image
        image[y_int, x_int] = (colors[:, :3] * 255).astype(np.uint8)
      
        return image

    server = viser.ViserServer(port=8080, verbose=False)
    _ = nerfview.Viewer(
        server=server,
        render_fn=viewer_render_fn,
        mode="rendering",
    )
    print("Viewer running... Ctrl+C to exit.")
    time.sleep(100000)


if __name__ == "__main__":
    main()