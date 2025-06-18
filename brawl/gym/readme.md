bash launch_servers.sh --env_name webarena --num_servers 2 --base_port 8000

# 2. (optional) start servers only, run Ray driver by hand
bash launch_servers.sh --only_servers &
export AGENTGYM_SERVER_BASE="http://localhost:8000"
CUDA_VISIBLE_DEVICES=0 python tools/grpo_autoscale.py --n_collectors 32