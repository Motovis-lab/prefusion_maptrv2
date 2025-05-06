from mtv4d import read_txt, read_pickle, write_pickle
import argparse
from pathlib import Path as P


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--pkl', default='/ssd1/MV4D_12V3L/fb_train_104_old_version.pkl')
    parser.add_argument('--txt', default='/ssd1/MV4D_12V3L/valset_indice.txt')
    parser.add_argument('--output', default='/tmp/output_dbg.pkl')  # /ssd1/MV4D_12V3L/valset_104.pkl 
    args = parser.parse_args()
    a = read_pickle(args.pkl)
    b = read_txt(args.txt)

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
    P(args.output).parent.mkdir(exist_ok=True, parents=True)
    write_pickle(output, args.output)
