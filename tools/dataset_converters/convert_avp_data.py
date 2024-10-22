import mmengine
from pathlib import Path as P


with open("data/avp/train.txt", 'r') as f:
    data = f.readlines()

with open("data/avp/val.txt", 'r') as f:
    data += f.readlines()

dump_data = dict(
    avp=dict(
        frame_info=dict()
    )
)
for idx, line in enumerate(data):
    line = line.strip()
    frame_id = str(idx).zfill(8)
    dump_data['avp']['frame_info'][frame_id] = dict(
        camera_image={"VCAMERA_FISHEYE_FRONT": P('AVP_raw_dataset') / P(line+".jpg")},
        camera_image_seg={"VCAMERA_FISHEYE_FRONT": P('AVP_raw_dataset') / P('mask') / P(line+".png")},
        camera_image_depth={"VCAMERA_FISHEYE_FRONT": None}
    )

mmengine.dump(dump_data, "data/avp/avp_train.pkl")