
import time
import gc

from tqdm import tqdm
from mmengine.config import Config
from prefusion.registry import RUNNERS
from mmengine.runner import Runner 




def get_data(dataloader):
    
    total_size = len(dataloader)
    for group_idx, group_batch in tqdm(enumerate(dataloader), total=total_size):
        for frame_idx, frame_batch in enumerate(group_batch):
            idx = group_idx * 1 + frame_idx
            time.sleep(0.01)
            # print(idx, frame_batch['index_infos'])
        gc.collect()
    print('DONE!')
            


if __name__ == '__main__':

    cfg = Config.fromfile('contrib/fastray_planar/configs/fastray_planar_single_frame_park_apa_scaled_relu6_a800.py')
    runner = Runner.from_cfg(cfg)
    runner.logger.name = "prefusion"
    
    print('Please start htop, and observe the RAM usage!')
    dataloader = Runner.build_dataloader(runner._train_dataloader)

    for _ in range(5):
        get_data(dataloader)
