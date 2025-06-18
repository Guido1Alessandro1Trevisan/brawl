# brawl/gym/policies/vllm_policy.py

import copy
import gc
from pathlib import Path
from typing import List

import GPUtil
import ray 
import torch
from vllm import LLM, SamplingParams

from brawl.gym.policies.base_policy import BasePolicy

# Goal for the future: Spin up a second VLLMPolicy actor with the new weights while the first keeps serving traffic; once ready, atomically update the reference held in the driver/collectors and kill the old actor.
@ray.remote(num_gpus=1)
class VLLMPolicy(BasePolicy):

    def __init__(self, model_id: str, **gen_cfg):
        self._model_id = model_id
        self._base_params = SamplingParams(**gen_cfg)
        self._init_llm(model_id)
    
    def _init_llm(self, model_path=str):
        self.llm = LLM(model=model_path)

    async def act(self, prompt: str, k: int = 4) -> List[str]:
        # Is there a better way to copy / hot-reload params?
        params = copy.deepcopy(self._base_params)
        params.n = k
        outs = self.llm.generate([prompt], params)
        return [o.text.strip() for o in outs[0].outputs]
    
    async def reload(self, hf_folder: str | None = None):

        target = Path(hf_folder or self._model_id).expanduser()

        del self.llm
        gc.collect()
        torch.cuda.empty_cache()

        self._init_llm(str(target))

        self._model_id = (str(target))

    async def get_gpu_util(self) -> float:
        gpu = GPUtil.getGPUs()[0]
        return gpu.load

