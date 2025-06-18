# brawl/gym/workers/collector.py
# workers/collector.py
from __future__ import annotations

import asyncio
import time
from typing import Dict, List

import ray
from omegaconf import DictConfig
from ray.util.queue import Queue, Empty, Full

from envs import ENV_REGISTRY
from policies.base_policy import BasePolicy


@ray.remote(num_cpus=1)
class Collector:  # not protocol – concrete Ray actor
    """Async rollout worker with gentle back‑pressure."""

    def __init__(
        self,
        cfg: DictConfig,
        policy: BasePolicy,
        traj_q: Queue,
    ):
        env_cls = ENV_REGISTRY[cfg.env.name]
        self.env = env_cls.options(num_cpus=1).remote(headless=cfg.env.headless)
        self.policy = policy
        self.k = cfg.policy.k
        self.q = traj_q
        self.last_ts = time.time()

    async def run_one(self, task_idx: int = 0) -> float:
        await self.env.reset.remote(seed=task_idx)
        prompt = await self.env.observe.remote()

        actions: List[str] = await self.policy.act.remote(prompt, self.k)

        for act in actions:
            s2, r, done = await self.env.step.remote(act)
            sample: Dict = dict(prompt=prompt, action=act, reward=r, done=done)
            # put with back‑pressure (≤100 ms per attempt)
            while True:
                try:
                    self.q.put(sample, timeout=0.1)
                    break
                except Full:
                    await asyncio.sleep(0.05)

            if done:
                await self.env.reset.remote(seed=task_idx)

        # FPS stat for optional logging
        now = time.time()
        fps = 1.0 / max(now - self.last_ts, 1e-6)
        self.last_ts = now
        return fps
