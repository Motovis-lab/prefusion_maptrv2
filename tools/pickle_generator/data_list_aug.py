import numpy as np
import mmengine
from mmengine import Config
config_file = "tools/evaluator/lidar.py"
cfg = Config.fromfile(config_file)
pkl_path = cfg.train_loader.dataset.info_path
a = mmengine.load(pkl_path)
# tongjishuju