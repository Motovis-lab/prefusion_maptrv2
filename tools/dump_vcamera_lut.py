import numpy as np
import virtual_camera as vc
import matplotlib.pyplot as plt

from pathlib import Path
from scipy.spatial.transform import Rotation

def load_calpara(file_path):
    with open(file_path, 'r') as f:
        extr_motovis = {}
        intr_motovis = {}
        for line in f:
            line_list = line.strip().split()
            if len(line_list) == 7:
                extr_motovis[line_list[0]] = np.array(line_list[1:], dtype=np.float64)
            if len(line_list) == 9:
                intr_motovis[line_list[0]] = np.array(line_list[1:], dtype=np.float64)
        return extr_motovis, intr_motovis


def get_extrinsic_from_calpara(extr_motovis):
    h, w, z, pitch, roll, yaw = extr_motovis
    R = Rotation.from_euler('xyz', (pitch, roll, yaw), degrees=False).as_matrix()
    print(Rotation.from_euler('xyz', (pitch, roll, yaw), degrees=False).as_euler('xyz', degrees=True))
    t = np.float32([h, w, z])
    return R, t

def get_intrinsic_from_calpara(intr_motovis):
    p0, p1, p2, p3, cx, cy, fx, fy = intr_motovis
    return cx, cy, fx, fy, p0, p1, p2, p3


extr_motovis, intr_motovis = load_calpara('./calpara.txt')


# extr_motovis = [-0.006110, 2.398196, 0.553211, -1.829800, 0.003799, -0.009039]
# intr_motovis = [0.275104, -0.044522, -0.000547, -0.000386, 956.428772, 639.237000, 372.554535, 372.701721]

vcam_params = {
    'Front': [(640, 384), (-120, 0, 0), (0, 2.5, 0.5)],
    'Left': [(640, 384), (-135, 0, 90), (-1, 0.5, 1)],
    'Right': [(640, 384), (-135, 0, -90), (1, 0.5, 1)],
    'Rear': [(640, 384), (-120, 0, 180), (0, -2.5, 0.5)],
}

for cam_id in extr_motovis:
    camera_real = vc.FisheyeCamera(
        resolution=(1920, 1280),
        extrinsic=get_extrinsic_from_calpara(extr_motovis[cam_id]),
        intrinsic=get_intrinsic_from_calpara(intr_motovis[cam_id]),
    )
    src_img = plt.imread(f'./png/{cam_id}.png')
    plt.imshow(src_img); plt.show()
    camera_virtual = vc.create_virtual_fisheye_camera(*vcam_params[cam_id])
    dst_img, _, uu, vv = vc.render_image(src_img, camera_real, camera_virtual, dump_uu_vv=True)
    plt.imshow(dst_img); plt.show()
    plt.imsave(f'vcam_{cam_id.lower()}.png', dst_img)
    with open(f'uu_vv_{cam_id.lower()}.txt', 'w') as f:
        for u in uu:
            f.write(f'{u:.3f} ')
        f.write('\n')
        for v in vv:
            f.write(f'{v:.3f} ')
        f.write('\n')
    