
import subprocess
from pathlib import Path
from typing import List

import GPUtil
import ray 
from vllm import LLM, SamplingParams

from policies.base_policy import BasePolicy

@ray.remote(num_gpus=1)
class VLLMPolicy(BasePolicy):

    def __init__(self, model_id: str, **gen_cfg):
        self.llm = LLM(model=model_id)
        self.params = SamplingParams(**gen_cfg)

    async def act(self, prompt: str, k: int = 4) -> List[str]:
        self.params.n = k
        outs = self.llm.generate([prompt], self.params)
        return [o.text.strip() for o in outs[0].outputs]
    
    async def reload(self, hf_folder: str | None = None):

        if hf_folder:
            folder = Path(hf_folder).expanduser()
            self.llm.load_model(str(folder))
