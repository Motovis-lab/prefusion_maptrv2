from mtv4d import read_txt, read_pickle, write_pickle

a = read_pickle('/ssd1/MV4D_12V3L/fb_train_104_old_version.pkl')
b = read_txt('/ssd1/MV4D_12V3L/valset_indice.txt')

output = {}
for key in b:
    sid, ts = key.split('/')
    if sid not in output.keys():
        output[sid] = {
            'scene_info': a[sid]['scene_info'],
            "meta_info": a[sid]['meta_info'],
            "frame_info": {},
        }
    output[sid]['frame_info'][ts] = a[sid]['frame_info'][ts]

write_pickle(output, '/ssd1/MV4D_12V3L/valset_104.pkl')