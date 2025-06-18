
from pathlib import Path
from typing import Dict, List

import ray
import torch
from omegaconf import OmegaConf
from ray.util.queue import Queue, Empty
from verl.trainer.ppo.ray_trainer import RayPPOTrainer

from learners.base_learner import BaseLearner

# Why here are there 4 cpus?
@ray.remote(num_gpus=1, num_cpus=4)
class PPOLearner(BaseLearner):
    
