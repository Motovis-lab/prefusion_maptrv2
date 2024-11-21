
import torch
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path

from prefusion.dataset.tensor_smith import *

__all__ = [
    'draw_out_feats',
    'draw_aligned_voxel_feats',
    'get_bbox_3d',
    'get_parkingslot_3d',
    'draw_outputs'
]


def draw_out_feats(
        batched_input_dict, 
        camera_tensors_dict,
        pred_bbox_3d,
        pred_polyline_3d=None,
        pred_parkingslot_3d=None,
    ):

    fig, _ = plt.subplots(3, 10)
    fig.suptitle(batched_input_dict['index_infos'][0].scene_frame_id)
    for i, cam_id in enumerate(camera_tensors_dict):
        img = camera_tensors_dict[cam_id].detach().cpu().numpy()[0].transpose(1, 2, 0)[..., ::-1] * 255 + 128
        img = img.astype(np.uint8)
        plt.subplot(3, 10, i+1)
        plt.title(cam_id.replace('VCAMERA_', '').lower())
        plt.imshow(img)

    gt_seg = batched_input_dict['annotations']['bbox_3d']['seg'][0][0].detach().cpu()
    pred_seg = pred_bbox_3d['seg'][0][0].to(torch.float32).sigmoid().detach().cpu()
    plt.subplot(3, 10, 11)
    plt.imshow(gt_seg)
    plt.title('bbox_3d gt_seg')
    plt.subplot(3, 10, 12)
    plt.imshow(pred_seg)
    plt.title("bbox_3d pred_seg")
    
    gt_cen = batched_input_dict['annotations']['bbox_3d']['cen'][0][0].detach().cpu()
    pred_cen = pred_bbox_3d['cen'][0][0].to(torch.float32).sigmoid().detach().cpu()
    pred_cen *= (pred_seg > 0.5)
    plt.subplot(3, 10, 13)
    plt.imshow(gt_cen)
    plt.title("bbox_3d gt_cen")
    plt.subplot(3, 10, 14)
    plt.imshow(pred_cen)
    plt.title("bbox_3d pred_cen")
    
    gt_reg = batched_input_dict['annotations']['bbox_3d']['reg'][0][0].detach().cpu()
    pred_reg = pred_bbox_3d['reg'][0][0].to(torch.float32).detach().cpu()
    pred_reg *= (pred_seg > 0.5)
    plt.subplot(3, 10, 15)
    plt.imshow(gt_reg)
    plt.title("bbox_3d gt_reg")
    plt.subplot(3, 10, 16)
    plt.imshow(pred_reg)
    plt.title("bbox_3d pred_reg")
    
    if pred_polyline_3d is not None:
        gt_seg = batched_input_dict['annotations']['polyline_3d']['seg'][0][0].detach().cpu()
        pred_seg = pred_polyline_3d['seg'][0][0].to(torch.float32).sigmoid().detach().cpu()
        plt.subplot(3, 10, 17)
        plt.imshow(gt_seg)
        plt.title('polyline_3d gt_seg')
        plt.subplot(3, 10, 18)
        plt.imshow(pred_seg)
        plt.title("polyline_3d pred_seg")
        
        gt_reg = batched_input_dict['annotations']['polyline_3d']['reg'][0][0].detach().cpu()
        pred_reg = pred_polyline_3d['reg'][0][0].to(torch.float32).detach().cpu()
        pred_reg *= (pred_seg > 0.5)
        plt.subplot(3, 10, 19)
        plt.imshow(gt_reg)
        plt.title("polyline_3d gt_reg")
        plt.subplot(3, 10, 20)
        plt.imshow(pred_reg)
        plt.title("polyline_3d pred_reg")
    
    if pred_parkingslot_3d is not None:
        gt_seg = batched_input_dict['annotations']['parkingslot_3d']['seg'][0][1].detach().cpu()
        pred_mask = pred_parkingslot_3d['seg'][0][0].to(torch.float32).sigmoid().detach().cpu()
        pred_seg = pred_parkingslot_3d['seg'][0][1].to(torch.float32).sigmoid().detach().cpu()
        plt.subplot(3, 10, 21)
        plt.imshow(gt_seg)
        plt.title('parkingslot_3d gt_seg')
        plt.subplot(3, 10, 22)
        plt.imshow(pred_seg)
        plt.title("parkingslot_3d pred_seg")
        
        gt_cen = batched_input_dict['annotations']['parkingslot_3d']['cen'][0][0].detach().cpu()
        pred_cen = pred_parkingslot_3d['cen'][0][0].to(torch.float32).sigmoid().detach().cpu()
        pred_cen *= (pred_mask > 0.5)
        plt.subplot(3, 10, 23)
        plt.imshow(gt_cen)
        plt.title("parkingslot_3d gt_cen")
        plt.subplot(3, 10, 24)
        plt.imshow(pred_cen)
        plt.title("parkingslot_3d pred_cen")
        
        gt_reg = batched_input_dict['annotations']['parkingslot_3d']['reg'][0][2].detach().cpu()
        pred_reg = pred_parkingslot_3d['reg'][0][2].to(torch.float32).detach().cpu()
        pred_reg *= (pred_mask > 0.5)
        plt.subplot(3, 10, 25)
        plt.imshow(gt_reg)
        plt.title("parkingslot_3d gt_reg")
        plt.subplot(3, 10, 26)
        plt.imshow(pred_reg)
        plt.title("parkingslot_3d pred_reg")
    

    voxel_range=([-0.5, 2.5], [36, -12], [12, -12])
    plt.subplot(3, 10, 27)
    plt.xlim(voxel_range[2])
    plt.ylim(voxel_range[1][::-1])
    
    gt_boxes_3d = batched_input_dict['transformables'][0]['bbox_3d']
    
    for element in gt_boxes_3d.elements:
        center = element['translation'][:, 0]
        xvec = element['size'][0] * element['rotation'][:, 0]
        yvec = element['size'][1] * element['rotation'][:, 1]
        corner_points = np.array([
            center + 0.5 * xvec - 0.5 * yvec,
            center + 0.5 * xvec + 0.5 * yvec,
            center - 0.5 * xvec + 0.5 * yvec,
            center - 0.5 * xvec - 0.5 * yvec
        ], dtype=np.float32)
        # print('gt: ', corner_points[:, :2])
        plt.plot(corner_points[[0, 1, 2, 3, 0], 1], corner_points[[0, 1, 2, 3, 0], 0], 'g')
    
    # gt_boxes_3d = batched_input_dict['annotations']['bbox_3d']
    plt.subplot(3, 10, 28)
    plt.xlim(voxel_range[2])
    plt.ylim(voxel_range[1][::-1])
    pred_bbox_3d_0 = {
        'cen': pred_bbox_3d['cen'][0].cpu().float().sigmoid(),
        'seg': pred_bbox_3d['seg'][0].cpu().float().sigmoid(),
        'reg': pred_bbox_3d['reg'][0].cpu().float()
    }
    pred_boxes_3d = get_bbox_3d(pred_bbox_3d_0)
    
    for element in pred_boxes_3d:
        # if element['confs'][0] < 0.7:
        #     continue
        if element['area_score'] < 0.5:
            continue
        center = element['translation']
        xvec = element['size'][0] * element['rotation'][:, 0]
        yvec = element['size'][1] * element['rotation'][:, 1]
        corner_points = np.array([
            center + 0.5 * xvec - 0.5 * yvec,
            center + 0.5 * xvec + 0.5 * yvec,
            center - 0.5 * xvec + 0.5 * yvec,
            center - 0.5 * xvec - 0.5 * yvec
        ], dtype=np.float32)
        # print('pred: ', corner_points[:, :2])
        plt.text(center[1], center[0], '{:.2f}'.format(element['area_score']), color='r',
                 ha='center', va='center')
        plt.plot(corner_points[[0, 1, 2, 3, 0], 1], corner_points[[0, 1, 2, 3, 0], 0], 'r')
    
    plt.subplot(3, 10, 29)
    plt.xlim(voxel_range[2])
    plt.ylim(voxel_range[1][::-1])

    gt_slots_3d = batched_input_dict['transformables'][0]['parkingslot_3d']
    
    for element in gt_slots_3d.elements:
        points = element['points']
        plt.plot(points[[1, 2, 3, 0], 1], points[[1, 2, 3, 0], 0], 'g')
    
    
    plt.subplot(3, 10, 30)
    plt.xlim(voxel_range[2])
    plt.ylim(voxel_range[1][::-1])
    
    pred_parkingslot_3d_0 = {
        'cen': pred_parkingslot_3d['cen'][0].cpu().float().sigmoid(),
        'seg': pred_parkingslot_3d['seg'][0].cpu().float().sigmoid(),
        'reg': pred_parkingslot_3d['reg'][0].cpu().float()
    }
    
    pred_slots_3d = get_parkingslot_3d(pred_parkingslot_3d_0)
    # print(pred_slots_3d)
    for slot in pred_slots_3d:
        # plt.text(points[0, 1], points[0, 0], '{:.2f}'.format(element['confs'][0]), color='r')
        plt.plot(slot[[1, 2, 3, 0], 1], slot[[1, 2, 3, 0], 0], 'r')
    
    plt.show()


def draw_aligned_voxel_feats(aligned_voxel_feats):

    n_frames = len(aligned_voxel_feats)
    plt.subplots(1, n_frames)
    for i, voxel_feats_frame in enumerate(aligned_voxel_feats):
        plt.subplot(1, n_frames, i+1)
        plt.imshow(voxel_feats_frame[0, 0].detach().cpu().to(torch.float32).numpy() > 0)
        plt.title(f'frame t({0 - i})')
    plt.show()


def get_bbox_3d(tensor_dict):
    from prefusion.dataset.tensor_smith import PlanarBbox3D
    pbox = PlanarBbox3D(
        voxel_shape=(6, 320, 160),
        voxel_range=([-0.5, 2.5], [36, -12], [12, -12]),
        reverse_pre_conf=0.3,
        reverse_nms_ratio=1.0
    )
    return pbox.reverse(tensor_dict)
    

def get_parkingslot_3d(tensor_dict):
    from prefusion.dataset.tensor_smith import PlanarParkingSlot3D
    pslot = PlanarParkingSlot3D(
        voxel_shape=(6, 320, 160),
        voxel_range=([-0.5, 2.5], [36, -12], [12, -12]),
        reverse_pre_conf=0.5
    )
    return pslot.reverse(tensor_dict)


def draw_outputs(pred_dict, batched_input_dict):

    camera_tensors_dict = batched_input_dict['camera_tensors']
    gt_dict = batched_input_dict['annotations']
    transformables = batched_input_dict['transformables'][0]
    scene_frame_id = batched_input_dict['index_infos'][0].scene_frame_id

    ncols = len(camera_tensors_dict)
    nrows = len(pred_dict) + 1

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2, nrows * 4))
    fig.suptitle(f'scene_frame_id: {scene_frame_id}')

    # plot camera images
    for i, cam_id in enumerate(camera_tensors_dict):
        img = camera_tensors_dict[cam_id].detach().cpu().numpy()[0].transpose(1, 2, 0)[..., ::-1] * 255 + 128
        img = img.astype(np.uint8)
        axes[0, i].imshow(img)
        axes[0, i].set_title(cam_id.replace('VCAMERA_', '').lower())
    
    # plot preds with labels
    for irow, branch in enumerate(pred_dict):
        irow += 1
        pred_dict_branch = pred_dict[branch]
        gt_dict_branch = gt_dict[branch]
        # extract batch_0
        pred_dict_branch_0 = {**pred_dict_branch}
        for key in pred_dict_branch_0:
            if key in ['cen', 'seg']:
                pred_dict_branch_0[key] = pred_dict_branch[key][0].sigmoid().detach().cpu().float()
            else:
                pred_dict_branch_0[key] = pred_dict_branch[key][0].detach().cpu().float()
        tensor_smith = transformables[branch].tensor_smith
        voxel_range = tensor_smith.voxel_range
        axes[irow, 0].annotate(
            branch, xy=(-0.5, 0.5), xycoords='axes fraction', ha='right', va='center', rotation=90
        )
        # gt_seg
        axes[irow, 0].imshow(gt_dict_branch['seg'][0][0].detach().cpu().float())
        axes[irow, 0].set_title(f'gt_seg_0')
        # pred_seg
        axes[irow, 1].imshow(pred_dict_branch_0['seg'][0])
        axes[irow, 1].set_title(f'pred_seg_0')

        match tensor_smith:
            case PlanarBbox3D() | PlanarRectangularCuboid() | PlanarSquarePillar():
                # plot gt bboxes
                axes[irow, 2].set_aspect('equal')
                axes[irow, 2].set_xlim(voxel_range[2])
                axes[irow, 2].set_ylim(voxel_range[1][::-1])
                axes[irow, 2].set_title(f'gt_bboxes')
                for element in transformables[branch].elements:
                    center = element['translation'][:, 0]
                    xvec = element['size'][0] * element['rotation'][:, 0]
                    yvec = element['size'][1] * element['rotation'][:, 1]
                    corner_points = np.array([
                        center + 0.5 * xvec - 0.5 * yvec,
                        center + 0.5 * xvec + 0.5 * yvec,
                        center - 0.5 * xvec + 0.5 * yvec,
                        center - 0.5 * xvec - 0.5 * yvec
                    ], dtype=np.float32)
                    axes[irow, 2].plot(corner_points[[0, 1, 2, 3, 0], 1], 
                                       corner_points[[0, 1, 2, 3, 0], 0], 'g')
                # plot pred bboxes
                axes[irow, 3].set_aspect('equal')
                axes[irow, 3].set_xlim(voxel_range[2])
                axes[irow, 3].set_ylim(voxel_range[1][::-1])
                axes[irow, 3].set_title(f'pred_bboxes')
                results = tensor_smith.reverse(pred_dict_branch_0)
                for element in results:
                    if element['area_score'] < 0.3:
                        continue
                    center = element['translation']
                    xvec = element['size'][0] * element['rotation'][:, 0]
                    yvec = element['size'][1] * element['rotation'][:, 1]
                    corner_points = np.array([
                        center + 0.5 * xvec - 0.5 * yvec,
                        center + 0.5 * xvec + 0.5 * yvec,
                        center - 0.5 * xvec + 0.5 * yvec,
                        center - 0.5 * xvec - 0.5 * yvec
                    ], dtype=np.float32)
                    # axes[irow, 3].text(center[1], center[0], 
                    #                    '{:.2f}'.format(element['area_score'] * element['confs'][0]),
                    #                    color='r', ha='center', va='center')
                    axes[irow, 3].plot(corner_points[[0, 1, 2, 3, 0], 1], 
                                       corner_points[[0, 1, 2, 3, 0], 0], 'r')
                for icol in range(4, ncols):
                    axes[irow, icol].axis('off')
            case PlanarParkingSlot3D():
                # plot gt slots
                axes[irow, 2].set_aspect('equal')
                axes[irow, 2].set_xlim(voxel_range[2])
                axes[irow, 2].set_ylim(voxel_range[1][::-1])
                axes[irow, 2].set_title(f'gt_slots')
                for element in transformables[branch].elements:
                    points = element['points']
                    axes[irow, 2].plot(points[[1, 2, 3, 0], 1], points[[1, 2, 3, 0], 0], 'g')
                # plot pred slots
                axes[irow, 3].set_aspect('equal')
                axes[irow, 3].set_xlim(voxel_range[2])
                axes[irow, 3].set_ylim(voxel_range[1][::-1])
                axes[irow, 3].set_title(f'pred_slots')
                results = tensor_smith.reverse(pred_dict_branch_0)
                for points in results:
                    axes[irow, 3].plot(points[[1, 2, 3, 0], 1], points[[1, 2, 3, 0], 0], 'r')
                for icol in range(4, ncols):
                    axes[irow, icol].axis('off')
            
            case _:
                for icol in range(2, ncols):
                    axes[irow, icol].axis('off')
    
    
    # plt.tight_layout()
    save_path = Path('work_dirs/result_pngs') / f'{scene_frame_id}.png'
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path)
    plt.close()
    # plt.show()
                
