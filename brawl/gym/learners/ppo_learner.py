
from pathlib import Path
from typing import Dict, List

import ray
import torch
from omegaconf import OmegaConf
from ray.util.queue import Queue, Empty
from verl.trainer.ppo.ray_trainer import RayPPOTrainer

from brawl.gym.learners.base_learner import BaseLearner

# Why here are there 4 cpus?
@ray.remote(num_gpus=1, num_cpus=4)
class PPOLearner(BaseLearner):
    """Thin wrapper around verll PPO trainer - one minibatch per call"""

    def __init__(self, cfg_path: str, traj_q: Queue):
        self.cfg = OmegaConf.load(Path(cfg_path))
        self.trainer = RayPPOTrainer(config=self.cfg)
        self.q = traj_q
        self.batch: List[Dict] = []

    async def backlog_size(self) -> int:
        return len(self.batch) + self.q.qsize()
    
    async def add_rollouts(self, batch: List[Dict]) -> None:
        self.trainer.add_rollouts(batch) 

    
