
from typing import Protocol, Tuple 

class BaseEnvActor(Protocol):

    async def reset(self, seed: int | None = None) -> str:
        ...
    
    async def observe(self) -> str:
        ...

    async def step(self, raw_action: str) -> Tuple[str, float, bool]:
        ...

    async def close(self) -> None:
        ...
