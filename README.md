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

### AWS quotas and setup

The default `cluster.yaml` requests one `g5.2xlarge` GPU worker and up to
fifty `c6i.xlarge` CPU workers in addition to the head node. Make sure your
AWS account has quotas for these instance types before launching the cluster.
1. Request at least **1× g5.2xlarge** instance in your chosen region.
2. Request **1× m6i.large** and up to **50× c6i.xlarge** instances.
3. Create a placement group named `wa-rl-pg` and an IAM profile called
   `wa-rl-profile`.

Once the quotas are approved, run `python tools/start_cluster.py` to create the
cluster and launch the driver. The script automatically uploads your code and
starts the training run on the head node.

### Experiment tracking

Training metrics are logged to [Weights & Biases](https://wandb.ai). To enable
logging:

1. Install the `wandb` Python package (already listed in `pyproject.toml`).
2. Authenticate by running `wandb login` and pasting your API key when
   prompted, or export it directly:

 ```bash
  export WANDB_API_KEY="YOUR_API_KEY"
  ```

Alternatively create a `.env` file in the repository root with:

```bash
WANDB_API_KEY=YOUR_API_KEY
```

The driver loads this file automatically when starting.

Once logged in, metrics will appear under the project and experiment names
specified in `conf/ppo_browser.yaml`.

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
