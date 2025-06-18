


import shutil
import tempfile
from pathlib import Path

import ray
from webarena.browser_env import ScriptBrowserEnv, create_id_based_action

from env.base_env import BaseEnvActor

@ray.remote(num_cpus=1)
class WebArenaEnvActor(BaseEnvActor):

    def __init__(self, headless: bool = True):
        
        self._profile_dir = Path(tempfile.mkdtemp(prefix="wa_profile_"))
        self.env = ScriptBrowserEnv(
            headless=headless,
            slow_mo=100,
            user_data_dir=str(self._profile_dir),
            observation_type="accessibility_tree",
            current_viewport_only=True,
            viewport_size={"width": 1280, "height": 720}
        )
        self.state, _ = self.env.reset()

    async def observe(self) -> str:
        return self.state["text"]
    
    async def step(self, raw_action: str):

        # Figure out how to create action based on id
        obs, r, term, trunc, _ = self.env.step(create_id_based_action(raw_action))
        self.state = obs
        return obs["text"], float(r), bool(term or trunc)

    async def reset(self, seed: int | None = 0):
        self.state, _ = self.env.reset(seed=seed)
        return self.state["text"]

    async def close(self):
        try:
            self.env.close()
        finally:
            shutil.rmtree(self._profile_dir, ignore_errors=True)

    def __del__(self):
        ray.get(self.close.remote())