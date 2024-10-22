from prefusion.registry import DATASETS, MODELS
from prefusion.runner import GroupRunner
from prefusion.registry import RUNNERS
import matplotlib.pyplot as plt
import ipdb
from mmengine import Config
from mmseg.models import EncoderDecoder

cfg = Config.fromfile("contrib/pretrain/configs/mv_4d_fastbev_pretrain.py")

runner = RUNNERS.build(cfg)

dataloader = runner.train_dataloader
model = runner.model

for data in dataloader:
    # ipdb.set_trace()
    model(data)