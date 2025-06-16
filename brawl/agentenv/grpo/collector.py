
import time
from typing import Any, Dict, Optional, Type

import ray
from brawl.agentenv.controller import BaseEnvClient

@ray.remote(num_cpus=1)
class GRPOCollector:

    def __init__(self,
        env_cls: Type[BaseEnvClient],
        env_kwargs: Dict[str, Any],
        policy,
        k: int):
        self.env = env_cls(**env_kwargs)
        self.policy = policy
        self.k = k
        self.last_t = time.time()

    async def run_one(self, idx: int,
                      meta: Optional[Dict[str, Any]] = None):
        
        self.env.reset(idx)
        prompt = self.env.observe()

        # understand if this line works and is correct
        actions = await self.policy.act.remote(prompt, k=self.k)

        trajs = []
        for a in actions:
            s2, r, done = self.env.step(a)
            trajs.append(
                dict(
                    prompt=prompt,
                    response=a,
                    reward=float(r),
                    done=done,
                    meta=meta or {}
                )
            )

        now = time.time()
        hz = 1.0 / max(now - self.last_t, 1e-6)
        self.last_t = now
        return trajs, hz

