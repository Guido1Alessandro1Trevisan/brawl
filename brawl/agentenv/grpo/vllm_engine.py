
import ray, torch


@ray.remote(num_gpus=1)
class VLMEngine:
    """One GPU that serves inference requests for all collectors"""

    def __init__(self, model_id: str, gen_cfg: dict):

        self.llm = LLM
