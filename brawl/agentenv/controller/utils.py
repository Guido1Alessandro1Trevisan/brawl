# utils.py

from dataclasses import dataclass
from typing import Optional, Sequence, TypedDict, Literal, List

import numpy as np
from transformers import GenerationConfig

from brawl.agentenv.controller.agent import Agent
from brawl.agentenv.controller.task import BaseTask, ExperienceOutput


class ConversationMessage(TypedDict):
    from_: Literal["human", "gpt"]
    value: str
    loss: Optional[bool]


@dataclass
class EvaluationOutput:
    experiences: Sequence[ExperienceOutput]
    score: float
    success: float


class BaseAgentEnvController:
    def __init__(self, agent: Agent, tasks: Sequence[BaseTask]) -> None:
        self.agent = agent
        self.tasks = tasks


    def generate_experience(
        self,
        idxs: Sequence[int] | Sequence[Sequence[int]],
        generation_config: Optional[GenerationConfig] = None,
        max_rounds: Optional[int] = None,
    ) -> list[ExperienceOutput]:

        experience: list[ExperienceOutput] = []

        if isinstance(idxs[0], int):
            # single task, flat idx list
            experience.extend(
                self.tasks[0].generate_experience(
                    idxs,
                    generation_config=generation_config,
                    max_rounds=max_rounds,
                )
            )
        elif isinstance(idxs[0], Sequence):
            # one idx-list per task
            for task, task_idxs in zip(self.tasks, idxs):
                experience.extend(
                    task.generate_experience(
                        task_idxs,
                        generation_config=generation_config,
                        max_rounds=max_rounds,
                    )
                )
        else:
            raise ValueError("Incorrect format for idxs")

        return experience



class Evaluator(BaseAgentEnvController):
    def eval(
        self,
        generation_config: Optional[GenerationConfig] = None,
        max_rounds: Optional[int] = None,
        idxs: Sequence[int] | Sequence[Sequence[int]] | None = None,
    ) -> EvaluationOutput:

        if idxs is None:
            idxs = [
                list(range(len(task.env_clients[0]))) for task in self.tasks
            ]

        exps = self.generate_experience(
            idxs=idxs,
            generation_config=generation_config,
            max_rounds=max_rounds,
        )

        rewards = np.asarray([exp.reward for exp in exps], dtype=np.float32)
        return EvaluationOutput(
            experiences=exps,
            score=float(rewards.mean()),
            success=float((rewards == 1).mean()),
        )


class BaseTrainer(BaseAgentEnvController):
    def train(self) -> None:
        pass

    def eval(
        self,
        generation_config: Optional[GenerationConfig] = None,
        max_rounds: Optional[int] = None,
        idxs: Sequence[int] | Sequence[Sequence[int]] | None = None,
    ) -> EvaluationOutput:

        if idxs is None:
            idxs = [
                list(range(len(task.env_clients[0]))) for task in self.tasks
            ]

        exps = self.generate_experience(
            idxs=idxs,
            generation_config=generation_config,
            max_rounds=max_rounds,
        )

        rewards = np.asarray([exp.reward for exp in exps], dtype=np.float32)
        return EvaluationOutput(
            experiences=exps,
            score=float(rewards.mean()),
            success=float((rewards == 1).mean()),
        )

    def save_model(self) -> None:
        pass
