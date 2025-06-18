# brawl/gym/policies/base_policy.py

from typing import Protocol, List

class BasePolicy(Protocol):
    async def act(self, prompt: str, k: int = 4) -> List[str]:
        ...

    async def reload(self, hf_folder: str | None = None) -> None:
        ...

    async def get_gpu_util(self) -> float:
        ...
