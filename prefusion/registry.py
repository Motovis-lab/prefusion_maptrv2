
from mmengine.registry import DATA_SAMPLERS as MMENGINE_DATA_SAMPLERS
from mmengine.registry import DATASETS as MMENGINE_DATASETS
from mmengine.registry import EVALUATOR as MMENGINE_EVALUATOR
from mmengine.registry import FUNCTIONS as MMENGINE_FUNCTIONS
from mmengine.registry import HOOKS as MMENGINE_HOOKS
from mmengine.registry import INFERENCERS as MMENGINE_INFERENCERS
from mmengine.registry import LOG_PROCESSORS as MMENGINE_LOG_PROCESSORS
from mmengine.registry import LOOPS as MMENGINE_LOOPS
from mmengine.registry import METRICS as MMENGINE_METRICS
from mmengine.registry import MODEL_WRAPPERS as MMENGINE_MODEL_WRAPPERS
from mmengine.registry import MODELS as MMENGINE_MODELS
from mmengine.registry import \
    OPTIM_WRAPPER_CONSTRUCTORS as MMENGINE_OPTIM_WRAPPER_CONSTRUCTORS
from mmengine.registry import OPTIM_WRAPPERS as MMENGINE_OPTIM_WRAPPERS
from mmengine.registry import OPTIMIZERS as MMENGINE_OPTIMIZERS
from mmengine.registry import PARAM_SCHEDULERS as MMENGINE_PARAM_SCHEDULERS
from mmengine.registry import \
    RUNNER_CONSTRUCTORS as MMENGINE_RUNNER_CONSTRUCTORS
from mmengine.registry import RUNNERS as MMENGINE_RUNNERS
from mmengine.registry import TASK_UTILS as MMENGINE_TASK_UTILS
from mmengine.registry import TRANSFORMS as MMENGINE_TRANSFORMS
from mmengine.registry import VISBACKENDS as MMENGINE_VISBACKENDS
from mmengine.registry import VISUALIZERS as MMENGINE_VISUALIZERS
from mmengine.registry import \
    WEIGHT_INITIALIZERS as MMENGINE_WEIGHT_INITIALIZERS
from mmengine.registry import Registry

# manage all kinds of runners like `EpochBasedRunner` and `IterBasedRunner`
RUNNERS = Registry(
    # TODO: update the location when prefusion has its own runner
    'runner',
    parent=MMENGINE_RUNNERS,
    locations=['prefusion']
)
# manage runner constructors that define how to initialize runners
RUNNER_CONSTRUCTORS = Registry(
    'runner constructor',
    parent=MMENGINE_RUNNER_CONSTRUCTORS,
    # TODO: update the location when prefusion has its own runner
    locations=['prefusion']
)
# manage all kinds of loops like `EpochBasedTrainLoop`
LOOPS = Registry(
    # TODO: update the location when prefusion has its own loop
    'loop',
    parent=MMENGINE_LOOPS,
    locations=['prefusion.runner']
)
# manage all kinds of hooks like `CheckpointHook`
HOOKS = Registry(
    'hook', parent=MMENGINE_HOOKS, locations=['prefusion.hooks']
)
# manage data-related modules
DATASETS = Registry(
    'dataset', parent=MMENGINE_DATASETS, locations=['prefusion']
)
DATASET_TOOLS = Registry('dataset_tool')
DATA_SAMPLERS = Registry(
    'data sampler',
    parent=MMENGINE_DATA_SAMPLERS,
    # TODO: update the location when prefusion has its own data sampler
    locations=['prefusion']
)
# mangage all kinds of modules inheriting `nn.Module`
MODELS = Registry(
    'model', parent=MMENGINE_MODELS, locations=['prefusion']
)
# mangage all kinds of model wrappers like 'MMDistributedDataParallel'
MODEL_WRAPPERS = Registry(
    'model_wrapper',
    parent=MMENGINE_MODEL_WRAPPERS,
    locations=['prefusion']
)
# mangage all kinds of weight initialization modules like `Uniform`
WEIGHT_INITIALIZERS = Registry(
    'weight initializer',
    parent=MMENGINE_WEIGHT_INITIALIZERS,
    locations=['prefusion']
)
# mangage all kinds of optimizers like `SGD` and `Adam`
OPTIMIZERS = Registry(
    'optimizer',
    parent=MMENGINE_OPTIMIZERS,
    # TODO: update the location when prefusion has its own optimizer
    locations=['prefusion']
)
# manage optimizer wrapper
OPTIM_WRAPPERS = Registry(
    'optim wrapper',
    parent=MMENGINE_OPTIM_WRAPPERS,
    # TODO: update the location when prefusion has its own optimizer
    locations=['prefusion']
)
# manage constructors that customize the optimization hyperparameters.
OPTIM_WRAPPER_CONSTRUCTORS = Registry(
    'optimizer wrapper constructor',
    parent=MMENGINE_OPTIM_WRAPPER_CONSTRUCTORS,
    # TODO: update the location when prefusion has its own optimizer
    locations=['prefusion']
)
# mangage all kinds of parameter schedulers like `MultiStepLR`
PARAM_SCHEDULERS = Registry(
    'parameter scheduler',
    parent=MMENGINE_PARAM_SCHEDULERS,
    # TODO: update the location when prefusion has its own scheduler
    locations=['prefusion']
)
# manage all kinds of metrics
METRICS = Registry(
    'metric', parent=MMENGINE_METRICS, locations=['prefusion']
)
# manage evaluator
EVALUATOR = Registry(
    'evaluator', parent=MMENGINE_EVALUATOR, locations=['prefusion']
)
# manage task-specific modules like anchor generators and box coders
TASK_UTILS = Registry(
    'task util', parent=MMENGINE_TASK_UTILS, locations=['prefusion']
)
# manage visualizer
VISUALIZERS = Registry(
    'visualizer',
    parent=MMENGINE_VISUALIZERS,
    locations=['prefusion']
)
# manage visualizer backend
VISBACKENDS = Registry(
    'vis_backend',
    parent=MMENGINE_VISBACKENDS,
    locations=['prefusion']
)

# manage logprocessor
LOG_PROCESSORS = Registry(
    'log_processor',
    parent=MMENGINE_LOG_PROCESSORS,
    # TODO: update the location when prefusion has its own log processor
    locations=['prefusion']
)

# manage inferencer
INFERENCERS = Registry(
    'inferencer',
    parent=MMENGINE_INFERENCERS,
    locations=['prefusion']
)


FUNCTIONS = Registry(
    'function',
    parent=MMENGINE_FUNCTIONS,
    locations=['prefusion']
)


TRANSFORMS = Registry(
    'transform',
    parent=MMENGINE_TRANSFORMS,
    locations=['prefusion']
)

# base data struct that can be transformed
TRANSFORMABLES = Registry(
    'transformable',
    locations=['prefusion']
)

TRANSFORMABLE_LOADERS = Registry(
    'transformable_loader',
    locations=['prefusion']
)


TENSOR_SMITHS = Registry(
    'tensor_smith',
    locations=['prefusion']
)


# an alternative way for data preprocessing
MODEL_FEEDERS = Registry(
    'model_feeder',
    locations=['prefusion']
)