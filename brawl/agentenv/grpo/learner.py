
import ray, torch
from omegaconf import OmegaConf
from pathlib import Path
from verl.trainer.ppo.ray_trainer import RayPPOTrainer


# why here do you need num cpus to be 4?
@ray.remote(num_gpus=1, num_cpus=4)
class Learner:
    def __init__(self, cfg_path: str):
        cfg = OmegaConf.load(Path(cfg_path).expanduser())

        self.trainer = RayPPOTrainer(config=cfg)
        self.trainer.init_workers()

    def backlog_size(self) -> int:
        return getattr(self.trainer, "dataloader_prefetch_len", 0)
    
    def update_forever(self):
        while True:
            self.trainer.fit()
            with torch.no_grad(): torch.cuda.synchronize()

