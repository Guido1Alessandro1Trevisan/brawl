
import ray

from vllm import LLM, SamplingParams


@ray.remote(num_gpus=1)
class VLMEngine:
    """One GPU that serves inference requests for all collectors"""

    def __init__(self, model_id: str, gen_cfg: dict):

        self.llm = LLM(model=model_id)
        self.params = SamplingParams(**gen_cfg)

    def act(self, prompt: str, k: int):
        """Generates k completions
        Returns:
            list[str] length === k
        """ 
        self.params.n = k

        outs = self.llm.generate([prompt], self.params)
        return [c.text for c in outs[0].outputs]
