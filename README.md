# Brawl

Brawl provides tooling for reinforcement learning of large language models.  The
repository contains two main packages:

- **`brawl`** – lightweight RL utilities built on top of `ray` for
  browser‑based environments.  It includes:
  - **drivers** (`brawl/gym/drivers`) – an asynchronous driver that spawns
    policy, learner and rollout workers.
  - **env** (`brawl/gym/env`) – environment interfaces and a `WebArena` binding
    to interact with a browser.
  - **learners** (`brawl/gym/learners`) – currently a simple PPO learner built
    on top of VERL.
  - **policies** (`brawl/gym/policies`) – wrappers around vLLM models for
    generation.
  - **workers** (`brawl/gym/workers`) – rollout collectors communicating with
    the environment and pushing trajectories to the learner.

- **`verl`** – a vendor‑agnostic library containing model loaders,
  training utilities and single‑controller implementations.  It is a
  trimmed‑down copy of VERL used by the learner and policy classes.

The `pyproject.toml` lists the core dependencies such as `hydra`, `ray` and
`torch`.  A small helper script `setup.sh` installs the `uv` package manager
used to resolve locked dependencies stored in `uv.lock`.

## Running the async driver

```
# Start the WebArena environment and launch two workers listening on ports 8000+
bash launch_servers.sh --env_name webarena --num_servers 2 --base_port 8000

# Alternatively only start the servers and run the Ray driver manually
bash launch_servers.sh --only_servers &
export AGENTGYM_SERVER_BASE="http://localhost:8000"
CUDA_VISIBLE_DEVICES=0 python tools/grpo_autoscale.py --n_collectors 32
```

## Project layout

```
├── main.py                 # example entry point
├── brawl/                  # RL utilities
│   ├── gym/
│   │   ├── drivers/        # async driver
│   │   ├── env/            # environment abstractions
│   │   ├── learners/       # PPO learner
│   │   ├── policies/       # vLLM based policy
│   │   └── workers/        # collectors
└── verl/                   # model and training library
```

This repository assumes Python 3.12+.  After installing the dependencies you can
import the modules or run `main.py` which prints a greeting.
