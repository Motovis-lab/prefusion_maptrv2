from mmengine.runner.runner import Runner
from prefusion.registry import RUNNERS
from mmengine.dist import master_only
import os.path as osp

@RUNNERS.register_module()
class GroupRunner(Runner):
    
    @master_only
    def dump_config(self) -> None:
        """Dump config to `work_dir`."""
        if self.cfg.filename is not None:
            filename = osp.basename(self.cfg.filename)
        else:
            filename = f'{self.timestamp}.py'
        self.cfg.dump(osp.join(self.log_dir, filename))