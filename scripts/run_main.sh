
export PYTHONPATH=:${PYTHONPATH}

# Set parameters in configs/run_main.yaml file and run this script
base_config="run_main_configs.yaml"

srun --job-name=llama_1 --gres=gpu:a6000:1 --pty --time=3-0:00 python src/main.py \
      --base_configs=configs/${base_config}

#bash scripts/run_main.sh
