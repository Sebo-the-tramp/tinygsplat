""" My first Gaussian render
adapted from https://github.com/nerfstudio-project/gsplat/blob/main/examples/simple_viewer.py

"""

from tinygrad import Tensor, dtypes # Autograd library

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
from scipy.spatial.transform import Rotation as R
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
        
        # scale of every gaussian
        scale = np.array([1,1,1])
        
        # rotation of every gaussian
        rotation = np.array([0,0,0])        
        
        image = rasterize_gaussians(scale, rotation, viewmat, K, point_cloud_array_homogeneous, point_cloud_colors, width, height)            
                
        return image

    server = viser.ViserServer(port=8080, verbose=False)
    _ = nerfview.Viewer(
        server=server,
        render_fn=viewer_render_fn,
        mode="rendering",
    )
    print("Viewer running... Ctrl+C to exit.")
    time.sleep(100000)


def rasterize_gaussians(
    scale, 
    rotation, 
    viewmat, 
    K, 
    point_cloud_array_homogeneous, 
    point_cloud_colors, 
    width, 
    height
):
            
        print("c2w", viewmat[:3, :4])
        print("K", K)
        
        # fx = fy = 0.5 * width  
        # cx = width / 2
        # cy = height / 2
        
        # K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        
        P = np.dot(K, viewmat[:3, :4])
        
        r = R.from_euler('xyz', rotation, degrees=False)
        s = np.diag(scale)
                
        covariance_camera = np.array([r.as_matrix() @ s @ s @ r.as_matrix().T])
        covariance_camera = covariance_camera.repeat(point_cloud_array_homogeneous.shape[0], axis=0)        
        
        # points
        point_camera = np.dot(P, point_cloud_array_homogeneous.T).T

        z = point_camera[:, 2]

        # Get x and y coordinates
        x = point_camera[:, 0]
        y = point_camera[:, 1]

        means_x = np.array([x]) / (-z**2)
        means_y = np.array([y]) / (-z**2)

        means_stacked = np.stack((means_x, means_y), axis=1).T        

        # print(means_x)
        # print(means_y)
        eyes = np.array([np.eye(2)]).repeat(point_cloud_array_homogeneous.shape[0], axis=0)
        eyes = eyes / (-z.reshape(-1, 1, 1))
        
        J = np.concatenate((eyes, means_stacked), axis=2)
        # print(J.shape)
        # print(covariance_camera.shape)
        
        covariance_2d = J @ covariance_camera @ J.transpose(0, 2, 1)

        # Create mask for points inside the image boundaries
        mask = (x >= 0) & (x < width) & (y >= 0) & (y < height)

        # Apply mask to x, y, and colors
        x = x[mask]
        y = y[mask]
        colors = point_cloud_colors[mask]
        covariance_2d = covariance_2d[mask]        

        # Convert x and y to integer pixel indices
        x_int = x.astype(int)
        y_int = y.astype(int)
        
        batch_size = colors.shape[0]

        # Initialize the image
        image = np.ones((height, width, 3), dtype=np.uint8) * 255
        
        image = generate_splat(Tensor(np.stack((x, y), axis=1), dtype=dtypes.float32), Tensor(colors, dtype=dtypes.float32), Tensor(covariance_2d, dtype=dtypes.float32), (width, height))     
        
        return image.numpy()

        # # Render the points onto the image
        # image[y_int, x_int] = (colors[:, :3] * 255).astype(np.uint8)                
        
def generate_splat(coordinates, colors, covariance, img_size):
    kernel_size = img_size[0]
        
    W = img_size[0] # width of the image
    H = img_size[1] # height of the image
    batch_size = colors.shape[0] # number of Gaussians defined above

    inv_covariance, covariance_det = get_inverse_batched(covariance)

    x = np.linspace(-5, 5, kernel_size)
    y = np.linspace(-5, 5, kernel_size)
    yy, xx = Tensor(np.meshgrid(x, y))
    yy = yy.expand(batch_size, -1, -1)
    
    xx = xx.expand(batch_size, -1, -1)    

    xx = xx + coordinates[:,0].unsqueeze(1).unsqueeze(1) * 5
    yy = yy + coordinates[:,1].unsqueeze(1).unsqueeze(1) * 5

    xy = xx.stack(yy, dim=-1)
    z = Tensor.einsum('bxyi,bij,bxyj->bxy', xy, -0.5 * inv_covariance, xy)

    kernel = z.exp() / np.pi * covariance_det.sqrt().view(batch_size, 1, 1)
    kernel_max = kernel.max(axis=(1,2), keepdim=True) + 1e-6 # avoid division by zero
    kernel_norm = kernel / kernel_max 

    kernel_rgb = kernel_norm.unsqueeze(1).expand(-1, 3, -1, -1)
    rgb_values_reshaped = colors.unsqueeze(1).unsqueeze(1).permute(0,3,1,2)

    final_image_layers = rgb_values_reshaped * kernel_rgb    

    final_image = final_image_layers.sum(axis=0)

    final_image = final_image.clamp(0, 1)
    final_image = final_image.permute(1,2,0)

    return final_image

def get_inverse_batched(matrices):

    a = matrices[:,0,0]
    b = matrices[:,0,1]
    c = matrices[:,1,0]
    d = matrices[:,1,1]    

    det = a*d - b*c    

    d_new = d.unsqueeze(1)
    b_new = -b.unsqueeze(1)
    a_new = a.unsqueeze(1)
    c_new = -c.unsqueeze(1)

    top = d_new.cat(b_new, dim=1).unsqueeze(1)
    bottom = c_new.cat(a_new, dim=1).unsqueeze(1)

    inverse = top.stack(bottom, dim=2).reshape(-1,2,2) 
    inverse = inverse * (1.0 / det).unsqueeze(1).unsqueeze(1)

    return inverse, det

if __name__ == "__main__":
    main()