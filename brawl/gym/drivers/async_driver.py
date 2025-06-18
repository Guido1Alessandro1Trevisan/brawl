# drivers/async_driver.py
"""
Launch WebArena RL without lookup tables.

• Creates a single vLLM policy actor (GPU)
• Creates a PPOLearner actor (GPU) that pulls roll‑outs from a Ray Queue
• Spawns N Collector actors (CPU) that interact with the browser env
• Autoscaling logic adds/removes collectors based on GPU util & backlog
"""
from __future__ import annotations
import asyncio
from typing import List

import hydra
import ray
from omegaconf import DictConfig, OmegaConf
from ray.util.queue import Queue

# ← direct imports, no registries
from learners.ppo_learner import PPOLearner
from policies.vllm_policy import VLLMPolicy
from brawl.gym.workers.collector import Collector     # adjust if your path differs


# ──────────────────────────────────────────────────────────────────────────
async def driver(cfg: DictConfig) -> None:
    """Async part of the program (may use `await`)."""
    ray.init(address="auto")                          # local or head of cluster

    # 1. policy actor – one GPU
    policy = VLLMPolicy.options(num_gpus=1, name="policy").remote(
        cfg.policy.model_id, **cfg.policy.gen_cfg
    )

    # 2. learner + trajectory queue – one GPU
    traj_q  = Queue(maxsize=cfg.autoscale.queue_max)
    learner = PPOLearner.options(num_gpus=1).remote(cfg.learner.cfg_path, traj_q)

    # 3. spawn initial collectors (CPU actors)
    collectors: List[ray.actor.ActorHandle] = [
        Collector.options(num_cpus=1).remote(cfg, policy, traj_q)
        for _ in range(cfg.autoscale.init_collectors)
    ]

    # 3‑bis. background autoscaler
    async def autoscale() -> None:
        while True:
            await asyncio.sleep(cfg.autoscale.period_s)
            util    = await policy.get_gpu_util.remote()
            backlog = await learner.backlog_size.remote()

            if util < cfg.autoscale.target_gpu_util and backlog < cfg.autoscale.min_backlog:
                for _ in range(cfg.autoscale.step):
                    c = Collector.options(num_cpus=1).remote(cfg, policy, traj_q)
                    collectors.append(c)

            elif (
                util > cfg.autoscale.target_gpu_util
                and backlog > cfg.autoscale.max_backlog
                and len(collectors) > cfg.autoscale.step
            ):
                for _ in range(cfg.autoscale.step):
                    victim = collectors.pop(0)
                    ray.kill(victim, no_restart=True)

            print(f"GPU util {util:.1%} | backlog {backlog} | collectors {len(collectors)}")

    asyncio.create_task(autoscale())

    # 4. main training loop
    env_idx = 0
    while True:
        pending = [c.run_one.remote(env_idx + i) for i, c in enumerate(collectors)]
        await asyncio.gather(*pending)
        await learner.update_step.remote()
        env_idx = (env_idx + len(collectors)) % cfg.env.max_episodes


# ──────────────────────────────────────────────────────────────────────────
@hydra.main(version_base=None, config_path="../../conf", config_name="benchmark/webarena")
def main(cfg: DictConfig) -> None:
    """Hydra entry point (synchronous)."""
    print("Config\n------")
    print(OmegaConf.to_yaml(cfg, resolve=True))
    asyncio.run(driver(cfg))        # launch the async driver


if __name__ == "__main__":
    main()
