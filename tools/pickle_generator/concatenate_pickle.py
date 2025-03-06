from pathlib import Path as P
import mmengine
from tqdm import tqdm

pkl_paths = sorted(P('/ssd1/MV4D_12V3L').glob('planar_lidar*.pkl'))
print(len(pkl_paths))
output = {}
for path in tqdm(pkl_paths):
    print(path)
    a = mmengine.load(str(path))
    for k, v in a.items():
        output[k] = v
mmengine.dump(output, '/tmp/1234/planar_lidar_train.pkl')













