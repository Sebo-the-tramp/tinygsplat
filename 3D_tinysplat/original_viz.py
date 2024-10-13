import json
import hydra
import time
import numpy as np
import PIL.Image as Image

from omegaconf import DictConfig
import viser
import viser.transforms as tf
from viser import ViserServer
from open3d import *    
from hydra import initialize, compose
from scipy.spatial.transform import Rotation as R

import nerfview


def get_c2w(camera):
    c2w = np.eye(4, dtype=np.float32)
    c2w[:3, :3] = R.as_quat(camera.wxyz)
    c2w[:3, 3] = camera.position
    return c2w

def get_w2c(camera):
    c2w = get_c2w(camera)
    w2c = np.linalg.inv(c2w)
    return w2c

# Hydra main function
@hydra.main(version_base=None, config_path="./config/", config_name="config")
def main(cfg: DictConfig):

    # Load camera data from the JSON file
    with open(cfg.paths.json_path, "r") as file:
        camera_data = json.load(file)                

    # Initialize the Viser server
    server = viser.ViserServer()
    if cfg.share_url:
        server.request_share_url()

    @server.on_client_connect
    def _(client: viser.ClientHandle) -> None:        

        # Load the point cloud from the provided .ply file path
        point_cloud = io.read_point_cloud(cfg.paths.ply_path) # Read point cloud
        server.scene.add_point_cloud("point_cloud", 
                                    points=np.asarray(point_cloud.points), 
                                    colors=point_cloud.colors, 
                                    point_shape="circle", point_size=0.01)

        # Extract camera parameters and add cameras to the scene
        for frame in camera_data["frames"]:
            
            file_path = frame["file_path"]
            transform_matrix = np.array(frame["transform_matrix"])[:3,:3]
            
            # perform the conversion
            # rotate 180 degrees around the x-axis
            transform_matrix = np.dot(transform_matrix, np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]]))        
                
            transform_matrix = R.from_matrix(transform_matrix).as_quat()
            transform_matrix = transform_matrix[[3, 0, 1, 2]]
            
            position = np.array(frame["transform_matrix"])[:3,3]        
                    
            # read image 
            new_image_path = cfg.paths.images_path  + "_8/" + file_path.replace("images", "")
            image_array = np.array(Image.open(new_image_path))

            H, W = image_array.shape[:2]
            
            fy = camera_data["fl_y"]

            # Add camera with transformation matrix
            frustum = server.scene.add_camera_frustum(
                name="{}".format(file_path),
                # fov = 2 * np.arctan2(H / 2, fy),
                fov = 0.95,
                aspect = 1.77,
                scale = 0.05,
                image = image_array,
                color=[1, 0, 0],
                wxyz = transform_matrix,
                position=position, 
            )
            
            def attach_callback(
                frustum: viser.CameraFrustumHandle
            ) -> None:
                @frustum.on_click                
                def _(_):
                    T_world_current = tf.SE3.from_rotation_and_translation(
                        tf.SO3(client.camera.wxyz), client.camera.position
                    )
                    T_world_target = tf.SE3.from_rotation_and_translation(
                        tf.SO3(frustum.wxyz), frustum.position
                    ) @ tf.SE3.from_translation(np.array([0.0, 0.0, -0.01]))
                    
                    print(T_world_target)

                    T_current_target = T_world_current.inverse() @ T_world_target

                    for j in range(20):
                        T_world_set = T_world_current @ tf.SE3.exp(
                            T_current_target.log() * j / 19.0
                        )

                        # We can atomically set the orientation and the position of the camera
                        # together to prevent jitter that might happen if one was set before the
                        # other.
                        with client.atomic():
                            client.camera.wxyz = T_world_set.rotation().wxyz
                            client.camera.position = T_world_set.translation()

                        client.flush()  # Optional!
                        time.sleep(1.0 / 60.0)

                    # Mouse interactions should orbit around the frame origin.
                    client.camera.look_at = frustum.position
            
            attach_callback(frustum)     
            
        def update(self):
            if self.need_update:
                start = time.time()
                for client in self.server.get_clients().values():
                    image_array = np.array(Image.open(new_image_path))
                    
                    client.set_background_image(image_array, format="jpeg")
                    self.debug_idx += 1
                                        

    while True:
        
        time.sleep(0.5)

if __name__ == "__main__":
    main()
