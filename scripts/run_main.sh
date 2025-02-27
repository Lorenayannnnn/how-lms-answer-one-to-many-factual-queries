
export PYTHONPATH=:${PYTHONPATH}

# Set parameters in configs/run_main.yaml file and run this script
base_config="run_main_configs.yaml"

CUDA_VISIBLE_DEVICES=0 python src/main.py \
      --base_configs=configs/${base_config}

#bash scripts/run_main.sh
