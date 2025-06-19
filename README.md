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

The repository includes helper scripts to launch a Ray cluster on AWS.
After configuring `tools/cluster.yaml` run:

```bash
python tools/start_cluster.py
```

This will spin up the cluster, sync the code and execute:

```bash
python -m brawl.gym.drivers.async_driver benchmark=webarena
```

To run the driver locally without AWS simply execute the same command in
your Python environment.


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

## Large model checkpoints

Training with models reaching hundreds of billions of parameters requires ample
disk space and remote storage.  `tools/cluster.yaml` now provisions large EBS
volumes on the head and worker nodes.  Set `default_hdfs_dir` in
`conf/ppo_browser.yaml` to an S3 bucket (e.g. `s3://brawl-checkpoints`) and
enable FSDP parameter offloading to store massive checkpoints.
