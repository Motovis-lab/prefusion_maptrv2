
import time

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
            del frame_batch
            # print(f'{idx}/{total_size}')
    print('DONE!')
            


if __name__ == '__main__':

    cfg = Config.fromfile('contrib/fastray_planar/configs/debug.py')
    runner = Runner.from_cfg(cfg)
    runner.logger.name = "prefusion"
    
    print('Please start htop, and observe the RAM usage!')
    dataloader = Runner.build_dataloader(runner._train_dataloader)
    get_data(dataloader)
    