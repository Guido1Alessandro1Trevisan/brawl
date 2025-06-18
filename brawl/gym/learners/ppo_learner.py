# learners/ppo_learner.py
from __future__ import annotations
from pathlib import Path
from typing import List, Dict, Any

import ray
import torch
import numpy as np
from omegaconf import OmegaConf
from ray.util.queue import Queue, Empty
from transformers import AutoTokenizer


from verl import DataProto                              
from verl.trainer.ppo.ray_trainer import compute_response_mask, apply_kl_penalty, compute_advantage
from verl.trainer.ppo.ray_trainer import RayPPOTrainer

from .base_learner import BaseLearner



def _to_dataproto(samples: List[Dict[str, Any]], tok) -> DataProto:
    """Dicts → DataProto (minimal fields only)."""
    prompts, responses, rewards = zip(
        *[(s["prompt"], s["action"], s["reward"]) for s in samples]
    )
    enc_p = tok(list(prompts),   return_tensors="pt", padding=True)
    enc_r = tok(list(responses), return_tensors="pt", padding=True)

    return DataProto(
        batch={
            "input_ids"      : enc_p.input_ids,
            "attention_mask" : enc_p.attention_mask,
            "responses"      : enc_r.input_ids,
            "scalar_rewards" : torch.tensor(rewards, dtype=torch.float),
        },
        non_tensor_batch={"uid": np.arange(len(samples))},
    )



@ray.remote(num_gpus=1, num_cpus=4)
class PPOLearner(BaseLearner):                              # type: ignore[misc]
    """Wraps VERL’s RayPPOTrainer; does the dict→DataProto conversion here."""

    def __init__(self, cfg_path: str, traj_q: Queue):
        self.cfg = OmegaConf.load(Path(cfg_path))
        self.tr  = RayPPOTrainer(config=self.cfg)           # ← unchanged VERL code
        self.q   = traj_q
        self.buf: List[Dict] = []

        # one tokenizer on the learner process
        self.tok = AutoTokenizer.from_pretrained(
            self.cfg.policy.model_id, trust_remote_code=True
        )
        self.rollouts_seen = 0

    # ───── BaseLearner API ───── #
    async def backlog_size(self) -> int:
        return len(self.buf) + self.q.qsize()

    async def add_rollouts(self, batch: List[Dict]) -> None:
        self.buf.extend(batch)

    async def iterate(self) -> None:
        if not self.buf:
            return

        dp = _to_dataproto(self.buf, self.tok)
        dp.batch["response_mask"] = compute_response_mask(dp)

        # in‑reward KL (if enabled in YAML)
        if self.cfg.algorithm.use_kl_in_reward:
            dp, _ = apply_kl_penalty(
                dp,
                kl_ctrl   = self.tr.kl_ctrl_in_reward,
                kl_penalty= self.cfg.algorithm.kl_penalty,
            )
        else:
            dp.batch["token_level_rewards"] = dp.batch["scalar_rewards"].unsqueeze(1)

        # advantages / returns
        dp = compute_advantage(
            dp,
            adv_estimator = self.cfg.algorithm.adv_estimator,
            gamma         = self.cfg.algorithm.gamma,
            lam           = self.cfg.algorithm.lam,
            num_repeat    = 1,
            norm_adv_by_std_in_grpo = self.cfg.algorithm.get(
                "norm_adv_by_std_in_grpo", True
            ),
            multi_turn    = False,
            config        = self.cfg.algorithm,
        )

        # gradient steps
        if self.tr.use_critic:
            self.tr.critic_wg.update_critic(dp)

        if self.rollouts_seen >= self.cfg.trainer.critic_warmup:
            self.tr.actor_rollout_wg.update_actor(dp)

        self.rollouts_seen += len(self.buf)
        self.buf.clear()
        torch.cuda.synchronize()

    # convenience for driver
    async def update_step(self, max_items: int = 2048) -> int:
        pulled = 0
        for _ in range(max_items):
            try:
                self.buf.append(self.q.get_nowait())
                pulled += 1
            except Empty:
                break
        if not self.buf:
            return 0
        await self.iterate()
        return pulled
