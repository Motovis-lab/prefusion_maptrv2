import os
import shutil
from pathlib import Path 
import argparse

def copy_matching_folders(source_dir, target_dir, folder_names):
    for root, dirs, files in os.walk(source_dir):
        for dir_name in dirs:
            if dir_name in folder_names:
                source_path = os.path.join(root, dir_name)
                relative_path = os.path.relpath(source_path, source_dir)
                target_path = os.path.join(target_dir, relative_path)
                
                print(f"复制文件夹: {source_path} 到 {target_path}")
                shutil.copytree(source_path, target_path, dirs_exist_ok=True)

# 使用示例
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a 3D detector')
    parser.add_argument('scene_name', help='folder path')
    args = parser.parse_args()
    source_root = "/home/wuhan/mv_4d_data/"
    target_root = "/home/wuhan/146_data/"
    directory = args.scene_name
    folder_names = ["VCAMERA_FISHEYE_BACK", "VCAMERA_FISHEYE_FRONT", 
                    "VCAMERA_FISHEYE_LEFT",
                    "VCAMERA_FISHEYE_RIGHT",
                    "VCAMERA_PERSPECTIVE_BACK",
                    "VCAMERA_PERSPECTIVE_BACK_LEFT",
                    "VCAMERA_PERSPECTIVE_BACK_RIGHT",
                    "VCAMERA_PERSPECTIVE_FRONT",
                    "VCAMERA_PERSPECTIVE_FRONT_LEFT",
                    "VCAMERA_PERSPECTIVE_FRONT_RIGHT",
                    "ground",
                    "undistort_static_lidar1_model",
                    "occ",
                    "sdf",
                    "seg"
                    ]  # 替换为您想要复制的具体文件夹名字列表

    # 执行复制操作
    copy_matching_folders(Path(source_root) / Path(directory), Path(target_root) / Path(directory), folder_names)

    print(f"{directory} done !!!")
