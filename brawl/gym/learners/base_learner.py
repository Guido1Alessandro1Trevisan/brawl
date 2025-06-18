# brawl/gym/learners/base_learner.py

from typing import Protocol, List, Dict

class BaseLearner(Protocol):

    async def add_rollouts(self, batch: List[Dict]) -> None:
        ...

    async def iterate(self) -> None:
        ...

    async def backlog_size(self) -> int:
        ...
