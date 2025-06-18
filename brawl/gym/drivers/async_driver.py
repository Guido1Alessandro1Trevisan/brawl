# drivers/async_driver.py

import asyncio
from pathlib import Path
from typing import List

import ray
import hydra
from omegaconf import DictConfig, OmegaConf
from ray.util.queue import Queue


from learners import LEARNER_REGISTRY
from policies import POLICY_REGISTRY
from brawl.gym.workers.collector import Collector


@hydra.main(version_base=None, config_path="../conf", config_name="benchmark/webarena")
def main(cfg: DictConfig):
    print("Config\n------")
    print(OmegaConf.to_yaml(cfg, resolve=True))

    ray.init(address="auto")  # inside cluster OR local

    # 1.  policy
    pol_cls = POLICY_REGISTRY[cfg.policy.name]
    policy = pol_cls.options(num_gpus=1, name="policy").remote(
        cfg.policy.model_id, **cfg.policy.gen_cfg
    )

    # 2.  queue + learner
    traj_q = Queue(maxsize=cfg.autoscale.queue_max)
    learner_cls = LEARNER_REGISTRY[cfg.learner.name]
    learner = learner_cls.options(num_gpus=1).remote(cfg.learner.cfg_path, traj_q)

    # 3.  collectors
    collectors: List[ray.actor.ActorHandle] = [
        Collector.options(num_cpus=1).remote(cfg, policy, traj_q)
        for _ in range(cfg.autoscale.init_collectors)
    ]

    async def autoscale():
        """Simple reactive scaler based on GPU util and backlog."""
        while True:
            await asyncio.sleep(cfg.autoscale.period_s)
            util = await policy.get_gpu_util.remote()
            backlog = await learner.backlog_size.remote()

            if util < cfg.autoscale.target_gpu_util and backlog < cfg.autoscale.min_backlog:
                # scale up
                for _ in range(cfg.autoscale.step):
                    c = Collector.options(num_cpus=1).remote(cfg, policy, traj_q)
                    collectors.append(c)
            elif (
                util > cfg.autoscale.target_gpu_util
                and backlog > cfg.autoscale.max_backlog
                and len(collectors) > cfg.autoscale.step
            ):
                # scale down
                for _ in range(cfg.autoscale.step):
                    victim = collectors.pop(0)
                    ray.kill(victim, no_restart=True)

            print(
                f"GPU util {util:.1%} | backlog {backlog} | collectors {len(collectors)}"
            )

    # background scaler
    asyncio.create_task(autoscale())

    # 4.  main loop
    env_idx = 0
    while True:
        pending = [c.run_one.remote(env_idx + i) for i, c in enumerate(collectors)]
        await asyncio.gather(*pending)
        await learner.update_step.remote()
        env_idx = (env_idx + len(collectors)) % cfg.env.max_episodes


if __name__ == "__main__":
    main()
