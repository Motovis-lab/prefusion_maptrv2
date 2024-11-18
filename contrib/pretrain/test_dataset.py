from prefusion.registry import DATASETS, MODELS
from prefusion.runner import GroupRunner
from prefusion.registry import RUNNERS
import matplotlib.pyplot as plt
import ipdb
from mmengine import Config
from mmseg.models import EncoderDecoder
from mmseg.models import DepthwiseSeparableASPPHead
from mmseg.models import SegDataPreProcessor
from mmseg.models import CrossEntropyLoss
from mmengine.dataset import ConcatDataset
from torch.utils.data.dataset import ConcatDataset
from mmengine.dataset.sampler import DefaultSampler
from mmseg.models import FCNHead
from mmseg.models.decode_heads import decode_head
from mmseg.models.data_preprocessor import SegDataPreProcessor
from timm.models.vision_transformer import PatchEmbed, Block
from timm.models.vision_transformer import VisionTransformer

cfg = Config.fromfile("contrib/pretrain/configs/mv_4d_fastbev_pretrain.py")

runner = RUNNERS.build(cfg)

dataloader = runner.train_dataloader
model = runner.model

for data in dataloader:
    # ipdb.set_trace()
    model(data)