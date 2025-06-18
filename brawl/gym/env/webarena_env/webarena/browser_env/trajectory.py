# brawl/gym/env/webarena_env/webarena/browser_env/trajectory.py
from typing import Union

from .actions import Action
from .utils import StateInfo

Trajectory = list[Union[StateInfo, Action]]
