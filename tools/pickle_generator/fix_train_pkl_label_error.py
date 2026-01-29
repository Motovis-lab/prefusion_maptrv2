import pickle
import mmengine


def check_slot_label(polylines):
    for p in polylines:
        if p['class'] == 'class.parking.parking_slot':
            if len(p['points']) != 4:
                return False
    return True


def main(input_data):
    for scene in input_data.keys():
        b = a[scene]['frame_info']
        for ts in sorted(b.keys()):
            # check parking slot
            is_ps_valid = check_slot_label(b[ts]['3d_polylines'])
            if not is_ps_valid:
                b.pop(ts)
    return input_data


if __name__ == "__main__":
    a = mmengine.load('/ssd1/MV4D_12V3L/planar_lidar_nocamerapose_train.pkl')
    # a = mmengine.load('/ssd1/MV4D_12V3L/planar_lidar_nocamerapose_20231109_143452.pkl')
    output_data = main(a)
    mmengine.dump(output_data, '/ssd1/MV4D_12V3L/planar_lidar_nocamerapose_train_fix_label_error.pkl')









