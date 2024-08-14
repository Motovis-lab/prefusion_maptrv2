import cv2
import numpy as np
import virtual_camera as vc


INF_DIST = 1e8

def expand_line_2d(line, radius=1):
    vec = line[1] - line[0]
    norm_vec = vec / np.linalg.norm(vec)
    vert_norm_vec = norm_vec[::-1] * [1, -1]
    expand_vec = radius * vert_norm_vec
    point_0 = line[0] + expand_vec
    point_1 = line[0] - expand_vec
    point_2 = line[1] - expand_vec
    point_3 = line[1] + expand_vec
    return np.float32([point_0, point_1, point_2, point_3])


def _add_new_axis(arr, n):
    for _ in range(n):
        arr = arr[..., None]
    return arr

def vec_point2line_along_direction(point, line, direction):
    point = np.float32(point)
    try:
        num_extra_dim = [None,] * (len(point.shape) - 1)
        line = np.float32(line)[..., *num_extra_dim]
        vec = np.float32(direction)[..., *num_extra_dim]
    except:
        n_extra_dim = len(point.shape) - 1
        line = _add_new_axis(np.float32(line), n_extra_dim)
        vec = _add_new_axis(np.float32(direction), n_extra_dim)

    vec_l = line[1] - line[0]
    vec_p = line[1] - point
    C1 = vec[1] * vec_l[0] - vec[0] * vec_l[1]
    if np.abs(C1) < 1e-5:
        return np.full_like(point, np.inf)
    C2 = (vec[1] * vec_p[0] - vec[0] * vec_p[1]) / C1
    
    return vec_p - vec_l * C2
    

def dist_point2line_along_direction(point, line, direction):
    vec = vec_point2line_along_direction(point, line, direction)
    return np.linalg.norm(vec, axis=0)


def _sign(x):
    return 2 * (x > 0) - 1


def get_cam_type(name):
    if 'perspective' in name.lower():
        return 'PerspectiveCamera'
    elif 'fisheye' in name.lower():
        return 'FisheyeCamera'
    else:
        raise ValueError('Unknown camera type')
    

def get_voxel_points_in_ego(voxel_shape, voxel_range):
    # voxel_shape = (4, 320, 160)
    # voxel_range = [[-2, 2], [36, -12], [12, -12]]
    # output shape: (3, 4*320*160)
    # voxel_points = np.array([zz, xx, yy]).reshape(3, -1)
    Z, X, Y = voxel_shape

    fz = Z / (voxel_range[0][1] - voxel_range[0][0])
    fx = X / (voxel_range[1][1] - voxel_range[1][0])
    fy = Y / (voxel_range[2][1] - voxel_range[2][0])

    cz = - voxel_range[0][0] * fz - 0.5
    cx = - voxel_range[1][0] * fx - 0.5
    cy = - voxel_range[2][0] * fy - 0.5

    vzz, vxx, vyy = np.meshgrid(np.arange(Z), np.arange(X), np.arange(Y), indexing='ij')

    zz = (vzz - cz) / fz
    xx = (vxx - cx) / fx
    yy = (vyy - cy) / fy

    voxel_points = np.array([xx, yy, zz]).reshape(3, -1)

    return voxel_points


default_camera_feature_config = dict(
    ray_distance_num_channel=64,
    ray_distance_start=0.25,
    ray_distance_step=0.25,
    feature_downscale=8,
)


class VoxelLookUpTableGenerator:
    '''
    Generate LUTS from <x,y,z> to <img_id,u,v,d>.
    LUT = {
        \<cam_id\>: dict(
            uu=uu, 
            vv=vv, 
            dd=dd, 
            valid_map=valid_map,
            valid_map_sampled=valid_map_sampled,
            norm_density_map=norm_density_map
        )
    }
    '''
    def __init__(
            self,
            voxel_feature_config=dict(
                voxel_shape=(6, 320, 160),  # Z, X, Y in ego system
                voxel_range=([-0.5, 2.5], [36, -12], [12, -12]),
                ego_distance_max=40,
                ego_distance_step=5
            ),
            camera_feature_configs=dict(
                VCAMERA_FISHEYE_FRONT=default_camera_feature_config,
                VCAMERA_PERSPECTIVE_FRONT_LEFT=default_camera_feature_config,
                VCAMERA_PERSPECTIVE_BACK_LEFT=default_camera_feature_config,
                VCAMERA_FISHEYE_LEFT=default_camera_feature_config,
                VCAMERA_PERSPECTIVE_BACK=default_camera_feature_config,
                VCAMERA_FISHEYE_BACK=default_camera_feature_config,
                VCAMERA_PERSPECTIVE_FRONT_RIGHT=default_camera_feature_config,
                VCAMERA_PERSPECTIVE_BACK_RIGHT=default_camera_feature_config,
                VCAMERA_FISHEYE_RIGHT=default_camera_feature_config,
                VCAMERA_PERSPECTIVE_FRONT=default_camera_feature_config
            )
        ):
        '''
        inputs:
        - voxel_feature_config: dict
            - voxel_shape: (Z, X, Y) in ego system
            - voxel_range: ([zmin, zmax], [xmin, xmax], [ymin, ymax]) in ego system, in meters
            - ego_distance_max: maximum ego distance in meters
            - ego_distance_step: step size of ego distance
        - camera_feature_configs: dict
            - \<cam_id\>: dict
                - ray_distance_num_channel: number of channels in ray distance
                - ray_distance_start: start distance of ray in meters
                - ray_distance_step: step size of ray distance
                - feature_downscale: downscale of feature map
            
        '''
        self.voxel_feature_config = voxel_feature_config
        self.camera_feature_configs = camera_feature_configs
        self.voxel_shape = self.voxel_feature_config['voxel_shape']
        self.voxel_range = np.float32(self.voxel_feature_config['voxel_range'])
        # gen voxel_ego_points, in shape of (3, 4*320*160) <3, Z*X*Y>
        self.voxel_points = get_voxel_points_in_ego(self.voxel_shape, self.voxel_range)
    
    def generate(self, camera_images, seed=None):
        '''
        inputs:
        - camera_images: \<list of Image transformables\>
        - seed: seed for randomness of sampled valid_map

        outputs: 
        - LUT = {
            \<cam_id\>: dict(
                uu=uu, 
                vv=vv, 
                dd=dd, 
                valid_map=valid_map,
                valid_map_sampled=valid_map_sampled,
                norm_density_map=norm_density_map
            )
        }
        '''
        ego_distance_max = self.voxel_feature_config['ego_distance_max']
        ego_distance_step = self.voxel_feature_config['ego_distance_step']
        distances_ego = np.linalg.norm(self.voxel_points, axis=0)
        distance_bins = np.arange(0, ego_distance_max + ego_distance_step, ego_distance_step)

        LUT = {}

        # loop in cameras, gen LUTS from <x,y,z> to <img_id,u,v,d>
        keys = []
        density_maps = []
        for key in camera_images:
            keys.append(key)
            image_data = camera_images[key].data
            # gen camera points, in shape of (3, Z*X*Y)
            R_ec, t_ec = image_data['extrinsic']
            camera_points = R_ec.T @ (self.voxel_points - t_ec[..., None])
            # camera feature settings
            downscale = self.camera_feature_configs[key]['feature_downscale']
            resolution = np.array(image_data['img'].shape[:2][::-1]) // downscale
            intrinsics = np.array(image_data['intrinsic'])
            intrinsics[:4] /= downscale
            camera_class = getattr(vc, image_data['cam_type'])
            camera = camera_class(
                resolution,
                image_data['extrinsic'],
                intrinsics,
            )
            uu_float, vv_float = camera.project_points_from_camera_to_image(camera_points) # in shape of Z*X*Y
            # ray distance settings
            ray_distance_start = self.camera_feature_configs[key]['ray_distance_start']
            ray_distance_step = self.camera_feature_configs[key]['ray_distance_step']
            ray_distance_num_channel = self.camera_feature_configs[key]['ray_distance_num_channel']
            distances = np.linalg.norm(camera_points, axis=0) # Z*X*Y
            dd_float = (distances - ray_distance_start) / ray_distance_step
            # rasterize, Z*X*Y, 6*320*160
            uu = np.round(uu_float).astype(int)  
            vv = np.round(vv_float).astype(int)
            dd = np.round(dd_float).astype(int)
            dd[dd < ray_distance_num_channel] = ray_distance_num_channel - 1
            # get valid maps, Z*X*Y, 6*320*160
            valid_map = (uu >= 0) * (uu < resolution[0]) * (vv >= 0) * (vv < resolution[1]) * (camera_points[2] > 0)
            uv_mask = cv2.resize(image_data['ego_mask'], resolution)
            valid_map *= uv_mask[vv * valid_map, uu * valid_map].astype(bool)
            uu_float[~valid_map] = -1
            vv_float[~valid_map] = -1
            dd_float[~valid_map] = -1
            uu[~valid_map] = -1
            vv[~valid_map] = -1
            dd[~valid_map] = -1
            # allocate LUTS
            LUT[key] = dict(uu=uu, vv=vv, dd=dd, valid_map=valid_map)
            # gen voxel ray density for each camera, in shape of Z*X*Y
            density_map = np.zeros_like(distances_ego)
            for dist_ind in range(len(distance_bins) - 1):
                valid_dist = (distances_ego > distance_bins[dist_ind]) & (distances_ego <= distance_bins[dist_ind + 1])
                valid_u = uu_float[valid_dist * valid_map]
                valid_v = vv_float[valid_dist * valid_map]
                if len(valid_u) > 0 and len(valid_v) > 0:
                    density_map[valid_dist * valid_map] = (valid_u.max() - valid_u.min()) * (valid_v.max() - valid_v.min())
            density_maps.append(density_map)
            
        # normalize density
        density_maps = np.stack(density_maps)
        density_maps_norm = density_maps / (density_maps.sum(axis=0, keepdims=True) + 1e-5)
        
        # generate random sampled LUT from x,y,z to <cam_id> according to density
        rng = np.random.default_rng(seed)
        random_map = rng.random(distances_ego.shape)
        for key_ind, key in enumerate(keys):
            acc_density_min = density_maps_norm[:key_ind].sum(axis=0)
            acc_density_max = density_maps_norm[:key_ind + 1].sum(axis=0)
            LUT[key]['valid_map_sampled'] = (random_map >= acc_density_min) * (random_map < acc_density_max)
            LUT[key]['norm_density_map'] = density_maps_norm[key_ind]
        
        return LUT
        