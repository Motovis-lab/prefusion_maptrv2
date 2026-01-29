from typing import TYPE_CHECKING, Union

import torch
import numpy as np
from scipy.spatial.transform import Rotation as R
from mmdet3d.structures.bbox_3d.utils import limit_period
from copious.cv.geometry import points3d_to_homo, rt2mat

from prefusion.registry import MODEL_FEEDERS
from prefusion.dataset import BaseModelFeeder
from prefusion.dataset.transform import CameraImageSet, Bbox3D
from prefusion.utils.visualization import PointNotOnImageError, pinhole_project, corner_pts_to_img_bbox

__all__ = ["FrankenStreamPETRModelFeeder"]


if TYPE_CHECKING:
    from prefusion.dataset.transform import EgoPoseSet


@MODEL_FEEDERS.register_module()
class FrankenStreamPETRModelFeeder(BaseModelFeeder):
    """FrankenStreamPETRModelFeeder.

    Args
    ----
    Any: Any parameter or keyword arguments.
    """
    def __init__(self, *, visible_range=(-25.6, -25.6, -2, 25.6, 25.6, 2), lidar_extrinsics=None, bbox_3d_pos_repr="cuboid_center", **kwargs) -> None:
        super().__init__(**kwargs)
        self.visible_range = visible_range

        assert bbox_3d_pos_repr in ["cuboid_center", "bottom_center"]
        self.bbox_3d_pos_repr = bbox_3d_pos_repr

        # 4x4 transformation matrix from lidar to ego (i.e. what we normally provided in calib)
        self.T_e_l: Union[np.ndarray, None] = np.array(lidar_extrinsics) if lidar_extrinsics is not None else None

    def process(self, frame_batch: list) -> dict | list:
        """
        Process frame_batch, make it ready for model inputs

        Parameters
        ----------
        frame_batch : list
            list of input_dicts

        Returns
        -------
        processed_frame_batch: dict | list
        }
        """
        processed_frame_batch = []
        for frame in frame_batch:
            processed_frame = dict(index_info=frame["index_info"], **{t.name: t for t in frame["transformables"].values()})
            processed_frame["meta_info"] = {}
            for k, trnsfmb in processed_frame.items():
                if isinstance(trnsfmb, CameraImageSet):
                    cam_ids, camera_images = zip(*trnsfmb.transformables.items())
                    
                    # exchange the order of cameras (CAM_BACK_RIGHT <=> CAM_FRONT_LEFT) to align with the original implementation of StreamPETR
                    cam_ids = (cam_ids[0], cam_ids[1], cam_ids[5], cam_ids[3], cam_ids[4], cam_ids[2])
                    camera_images = (camera_images[0], camera_images[1], camera_images[5], camera_images[3], camera_images[4], camera_images[2])

                    img_tensor = torch.vstack(
                        [(t.tensor["img"] * t.tensor["ego_mask"]).unsqueeze(0) for t in camera_images]
                    )
                    processed_frame[k] = img_tensor
                    processed_frame["meta_info"][k] = {
                        "camera_ids": cam_ids,
                        "intrinsic": [self._intrinsic_param_to_4x4_mat(cam_im.intrinsic) for cam_im in camera_images],
                        "extrinsic": [rt2mat(*cam_im.extrinsic, as_homo=True) for cam_im in camera_images],
                        "extrinsic_inv": [np.linalg.inv(rt2mat(*cam_im.extrinsic, as_homo=True)) for cam_im in camera_images],
                    }
                    continue
                if isinstance(trnsfmb, Bbox3D):
                    if len(trnsfmb.tensor["classes"]) > 0:
                        visible_boxes = self.get_boxes_within_visible_range(trnsfmb.tensor)
                        processed_frame[k] = visible_boxes["xyz_lwh_yaw_vx_vy"]
                        processed_frame["meta_info"][k] = {"classes": visible_boxes["classes"]}
                        processed_frame["meta_info"][k].update(bbox3d_corners=visible_boxes["corners"])
                    else:
                        processed_frame[k] = trnsfmb.tensor["xyz_lwh_yaw_vx_vy"]
                        processed_frame["meta_info"][k] = {"classes": trnsfmb.tensor["classes"]}
                        processed_frame["meta_info"][k].update(bbox3d_corners=trnsfmb.tensor["corners"])

            processed_frame_batch.append(processed_frame)

        if self.T_e_l is not None:
            for frame in processed_frame_batch:
                frame["meta_info"]["T_ego_lidar"] = self.T_e_l
                self.derive_2d_bboxes_(frame) # must run before self.convert_model_food_to_lidar_coordsys_, coz' we need the boxes to be in ego coordsys
                self.convert_model_food_to_lidar_coordsys_(frame)

        # Dedicated data handling for Nuscenes data
        for frame in processed_frame_batch:
            frame.update(dictionary=frame_batch[0]['transformables']['bbox_3d'].dictionary)

        return processed_frame_batch

    @staticmethod
    def _intrinsic_param_to_4x4_mat(param):
        mat = np.eye(4)
        mat[0, 0] = param[2]
        mat[1, 1] = param[3]
        mat[0, 2] = param[0]
        mat[1, 2] = param[1]
        return mat

    def get_boxes_within_visible_range(self, box3d_tensor_dict: dict):
        box_centers = box3d_tensor_dict["xyz_lwh_yaw_vx_vy"][:, :3]
        visible_mask = (
            (box_centers[:, 0] >= self.visible_range[0]) & (box_centers[:, 0] <= self.visible_range[3])
            & (box_centers[:, 1] >= self.visible_range[1]) & (box_centers[:, 1] <= self.visible_range[4])
            & (box_centers[:, 2] >= self.visible_range[2]) & (box_centers[:, 2] <= self.visible_range[5])
        )
        return {k: v[visible_mask] for k, v in box3d_tensor_dict.items()}

    def convert_model_food_to_lidar_coordsys_(self, frame_data: dict):
        self.convert_ego_pose_set_to_lidar_coordsys_(frame_data["ego_poses"])
        self.convert_bbox3d_to_lidar_coordsys_(frame_data["bbox_3d"])
        self.convert_bbox3d_corners_to_lidar_coordsys_(frame_data["meta_info"]["bbox_3d"]["bbox3d_corners"])

    def convert_ego_pose_set_to_lidar_coordsys_(self, ego_pose_set: "EgoPoseSet"):
        if self.T_e_l is None:
            return
        for t, pose in ego_pose_set.transformables.items():
            # explaination: what we want is T_w_l = T_w_e @ T_e_l
            # `pose` is T_w_e
            pose.rotation, pose.translation = (
                pose.rotation @ self.T_e_l[:3, :3],
                pose.rotation @ self.T_e_l[:3, 3][:, None] + pose.translation
            )

    def convert_bbox3d_to_lidar_coordsys_(self, bbox3d: torch.Tensor):
        if self.T_e_l is None:
            return

        if len(bbox3d) == 0:
            return

        T_l_e = torch.tensor(np.linalg.inv(self.T_e_l), dtype=torch.float32)

        # convert positions from ego coordsys to lidar coordsys
        bbox3d[:, :3] = (T_l_e @ self.to_homo(bbox3d[:, :3]).T).T[:, :3]
        if self.bbox_3d_pos_repr == "bottom_center":
            bbox3d[:, 2] -= bbox3d[:, 5] / 2

        # convert rotations from ego coordsys to lidar coordsys
        rotations = R.from_euler("Z", bbox3d[:, 6].tolist(), degrees=False).as_matrix()
        yaws = R.from_matrix(T_l_e[:3, :3][None, :].numpy() @ rotations).as_euler("XYZ", degrees=False)[:, 2]
        bbox3d[:, 6] = torch.tensor(limit_period(yaws, offset=0.5, period=2 * np.pi), dtype=torch.float32) # limit rad to [-pi, pi]

        # convert velocity from ego coordsys to lidar coordsys (do not apply translation to velocity)
        bbox3d[:, 7:9] = (T_l_e[:2, :2] @ bbox3d[:, 7:9].T).T

    def convert_bbox3d_corners_to_lidar_coordsys_(self, bbox3d_corners: torch.Tensor):
        if self.T_e_l is None:
            return

        if len(bbox3d_corners) == 0:
            return

        T_l_e = torch.tensor(np.linalg.inv(self.T_e_l), dtype=torch.float32)

        N = bbox3d_corners.shape[0]
        bbox3d_corners[:, :, :3] = (T_l_e @ self.to_homo(bbox3d_corners.to(dtype=torch.float32).reshape(-1, 3)[:, :3]).T).T[:, :3].reshape(N, 8, 3)

    @staticmethod
    def to_homo(pts: torch.Tensor):
        return torch.cat((pts, torch.ones_like(pts[..., :1])), dim=-1)

    def derive_2d_bboxes_(self, frame_data: dict):
        cam_ids = frame_data['meta_info']['camera_images']['camera_ids']
        cam_extrinsics = frame_data['meta_info']['camera_images']['extrinsic']
        cam_intrinsics = frame_data['meta_info']['camera_images']['intrinsic']
        # T_w_e = rt2mat(frame_data["ego_poses"].transformables["0"].rotation, frame_data["ego_poses"].transformables["0"].translation, as_homo=True)
        bboxes_3d_corners = frame_data['meta_info']['bbox_3d']['bbox3d_corners']
        bboxes_3d = frame_data['bbox_3d']
        classes_3d = frame_data['meta_info']['bbox_3d']['classes']
        frame_data["bbox_2d"] = []
        frame_data["bbox_center_2d"] = []
        frame_data["meta_info"]["bbox_2d"] = {"classes": []}
        prefix = "dummy_intr"
        for cam_id, cam_extr, cam_intr, im in zip(cam_ids, cam_extrinsics, cam_intrinsics, frame_data['camera_images']):
            im_size = im.shape[-2:][::-1]
            bboxes_2d = []
            centers_2d = []
            classes_2d = []
            _extr = (cam_extr[:3, :3], cam_extr[:3, 3])
            _intr = (cam_intr[0, 2], cam_intr[1, 2], cam_intr[0, 0], cam_intr[1, 1])
            for bx_corners, bx, cls in zip(bboxes_3d_corners, bboxes_3d, classes_3d):
                try:
                    # Use conservative=False and filter_outside_image=False to keep all projected corners
                    # This allows corner_pts_to_img_bbox to properly handle the intersection
                    corners_on_img = pinhole_project(points3d_to_homo(bx_corners), _extr, _intr, im_size, conservative=False, filter_outside_image=False)
                except PointNotOnImageError:
                    # Skip only if ALL corners are behind the camera
                    continue
                
                # Always try to form 2D bbox from projected corners
                bbox2d = corner_pts_to_img_bbox(corners_on_img, imsize=im_size)
                
                # Skip only if no intersection with image canvas (no visible part)
                if bbox2d is None:
                    continue
                
                # For center: try to project the 3D center, but if it's outside, use 2D bbox center
                try:
                    center2d = pinhole_project(points3d_to_homo(bx[:3][None, :]), _extr, _intr, im_size)[0]
                except PointNotOnImageError:
                    # If 3D center projects outside image, use center of the clipped 2D bbox
                    center2d = np.array([(bbox2d[0] + bbox2d[2]) / 2, (bbox2d[1] + bbox2d[3]) / 2])
                
                bboxes_2d.append(bbox2d)
                centers_2d.append(center2d)
                classes_2d.append(cls)

            # FIXME: Visualize 2D boxes
            # import cv2
            # import matplotlib.pyplot as plt
            # im = (im.clone().numpy().transpose(1, 2, 0) * np.array([58.395, 57.120, 57.375]) + np.array([123.675, 116.280, 103.530])).astype(np.uint8)
            # im = np.ascontiguousarray(im)

            # for _bbox2d, _center2d in zip(bboxes_2d, centers_2d):
            #     cv2.rectangle(im, (int(_bbox2d[0]), int(_bbox2d[1])), (int(_bbox2d[2]), int(_bbox2d[3])), (0, 255, 0), 2)
            #     cv2.circle(im, (int(_center2d[0]), int(_center2d[1])), 5, (255, 0, 0), -1)
            
            # plt.figure(figsize=(10, 10))
            # plt.imshow(im)
            # plt.axis('off')
            # plt.title(f"Camera {cam_id} 2D BBoxes")
            # plt.tight_layout()
            # plt.savefig(f"2dbbox_vis_{cam_id}.png")
            # plt.close()
            # FIXME

            # FIXME: Visualize 3D boxes
            # import matplotlib.pyplot as plt
            # import cv2
            # im_3d = (im.clone().numpy().transpose(1, 2, 0) * np.array([58.395, 57.120, 57.375]) + np.array([123.675, 116.280, 103.530])).astype(np.uint8)
            # im_3d = np.ascontiguousarray(im_3d)

            # # Draw 3D boxes by connecting the projected corners
            # for bx_corners, cls in zip(bboxes_3d_corners, classes_3d):
            #     try:
            #         # Project all 8 corners of the 3D box
            #         # FIXME: temp modify the camera intrinsic manually for debugging
            #         # cam_intr = np.array([[689.047,   0.   , 454.623],
            #         #                     [  0.   , 689.047,  83.492],
            #         #                     [  0.   ,   0.   ,   1.   ]])
            #         # FIXME:
            #         corners_2d = pinhole_project(points3d_to_homo(bx_corners), _extr, _intr, im_size, 
            #                                    conservative=False, filter_outside_image=False)
                    
            #         # Convert to integer coordinates for drawing
            #         corners_2d = corners_2d.astype(np.int32)
                    
            #         # Define the 12 edges of a 3D box (connecting the 8 corners)
            #         # Box corner indices: 0-3 are bottom face, 4-7 are top face
            #         edges = [
            #             # Bottom face edges
            #             (0, 1), (1, 2), (2, 3), (3, 0),
            #             # Top face edges  
            #             (4, 5), (5, 6), (6, 7), (7, 4),
            #             # Vertical edges connecting bottom and top
            #             (0, 4), (1, 5), (2, 6), (3, 7)
            #         ]
                    
            #         # Choose color based on class (you can customize this)
            #         colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), 
            #                  (255, 0, 255), (0, 255, 255), (128, 128, 128), (255, 128, 0)]
            #         color = colors[cls % len(colors)]
                    
            #         # Draw each edge of the 3D box
            #         for start_idx, end_idx in edges:
            #             start_pt = corners_2d[start_idx]
            #             end_pt = corners_2d[end_idx]
                        
            #             # Only draw if both points are within reasonable bounds
            #             if (0 <= start_pt[0] < im_size[0]*2 and 0 <= start_pt[1] < im_size[1]*2 and
            #                 0 <= end_pt[0] < im_size[0]*2 and 0 <= end_pt[1] < im_size[1]*2):
            #                 cv2.line(im_3d, tuple(start_pt), tuple(end_pt), color, 1)
                    
            #         # Draw corner points
            #         for corner in corners_2d:
            #             if 0 <= corner[0] < im_size[0]*2 and 0 <= corner[1] < im_size[1]*2:
            #                 cv2.circle(im_3d, tuple(corner), 3, color, -1)
                            
            #     except (PointNotOnImageError, ValueError, IndexError):
            #         # Skip boxes that can't be properly projected
            #         continue
            
            # # Save the 3D visualization
            # plt.figure(figsize=(12, 8))
            # plt.imshow(im_3d)
            # plt.axis('off')
            # plt.title(f"Camera {cam_id} - 3D Bounding Boxes")
            # plt.tight_layout()
            # plt.savefig(f"{prefix}_3dbbox_vis_{cam_id}.png", dpi=150, bbox_inches='tight')
            # plt.close()
            
            # print(f"✅ Saved 3D bbox visualization for camera {cam_id}")
            # FIXME

            frame_data["bbox_2d"].append(torch.tensor(bboxes_2d))
            frame_data["bbox_center_2d"].append(torch.tensor(np.array(centers_2d)))
            frame_data["meta_info"]["bbox_2d"]["classes"].append(torch.tensor(classes_2d))

        # FIXME: Visualize 3D boxes in Bird's Eye View
        # fig, ax = plt.subplots(figsize=(12, 12))
        
        # # Set up BEV coordinate system (looking down from above)
        # bev_range = 50  # meters in each direction
        # ax.set_xlim(-bev_range, bev_range)
        # ax.set_ylim(-bev_range, bev_range)
        # ax.set_aspect('equal')
        # ax.grid(True, alpha=0.3)
        # ax.set_xlabel('X (forward) [m]')
        # ax.set_ylabel('Y (left) [m]')
        # ax.set_title(f'Bird\'s Eye View - 3D Bounding Boxes (Camera {cam_id} perspective)')
        
        # # Color map for different classes
        # colors = ['red', 'green', 'blue', 'yellow', 'magenta', 'cyan', 'orange', 'purple']
        # class_names = ['car', 'truck', 'bus', 'trailer', 'construction_vehicle', 
        #                 'pedestrian', 'motorcycle', 'bicycle', 'traffic_cone', 'barrier']
        
        # # Draw each 3D box in BEV
        # for i, (bx_corners, bx, cls) in enumerate(zip(bboxes_3d_corners, bboxes_3d, classes_3d)):
        #     # Extract bottom face corners (indices 0-3) for BEV
        #     bottom_corners = bx_corners[:4]  # First 4 corners are bottom face
            
        #     # Create polygon from bottom corners (close the shape)
        #     corners_x = np.append(bottom_corners[:, 0], bottom_corners[0, 0])
        #     corners_y = np.append(bottom_corners[:, 1], bottom_corners[0, 1])
            
        #     # Choose color based on class
        #     color = colors[cls % len(colors)]
        #     class_name = class_names[cls] if cls < len(class_names) else f'class_{cls}'
            
        #     # Draw the box outline
        #     ax.plot(corners_x, corners_y, color=color, linewidth=2, 
        #             label=class_name if i == 0 or cls not in [prev_cls for prev_cls in classes_3d[:i]] else "")
            
        #     # Fill the box with transparent color
        #     ax.fill(corners_x, corners_y, color=color, alpha=0.2)
            
        #     # Draw center point
        #     center_x, center_y = bx[0], bx[1]
        #     ax.plot(center_x, center_y, 'o', color=color, markersize=6)
            
        #     # Draw orientation arrow (yaw direction)
        #     arrow_length = max(bx[3], bx[4]) / 2  # Half of max(length, width)
        #     yaw = bx[6]  # Yaw angle
        #     arrow_end_x = center_x + arrow_length * np.cos(yaw)
        #     arrow_end_y = center_y + arrow_length * np.sin(yaw)
        #     ax.arrow(center_x, center_y, arrow_end_x - center_x, arrow_end_y - center_y,
        #             head_width=1.0, head_length=1.0, fc=color, ec=color, alpha=0.8)
            
        #     # Add box ID text
        #     ax.text(center_x, center_y + 1, f'{i}', fontsize=8, ha='center', 
        #             bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))
        
        # # Draw ego vehicle at origin
        # ego_size = 2.0
        # ego_corners_x = [-ego_size/2, ego_size/2, ego_size/2, -ego_size/2, -ego_size/2]
        # ego_corners_y = [-ego_size/2, -ego_size/2, ego_size/2, ego_size/2, -ego_size/2]
        # ax.plot(ego_corners_x, ego_corners_y, 'k-', linewidth=3, label='Ego Vehicle')
        # ax.fill(ego_corners_x, ego_corners_y, color='black', alpha=0.3)
        
        # # Add legend (only unique classes)
        # handles, labels = ax.get_legend_handles_labels()
        # by_label = dict(zip(labels, handles))
        # ax.legend(by_label.values(), by_label.keys(), loc='upper right', bbox_to_anchor=(1.15, 1))
        
        # # Add statistics text
        # stats_text = f'Total boxes: {len(bboxes_3d)}\nVisible range: {self.visible_range}'
        # ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
        #         verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        # plt.tight_layout()
        # plt.savefig(f'{prefix}_bev_3dboxes.png', dpi=150, bbox_inches='tight')
        # plt.close()
        
        # print(f"✅ Saved BEV 3D bbox visualization")
        # # FIXME

        # a = 100