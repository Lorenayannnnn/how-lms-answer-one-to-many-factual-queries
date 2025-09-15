
export PYTHONPATH=:${PYTHONPATH}

PROJECT_ROOT_DIR=.
EXAMINE_TYPE="noise_subject"    # noise_subject or noise_prev_ans
MODEL_NAME="meta-llama/Meta-Llama-3-8B-Instruct"  # meta-llama/Meta-Llama-3-8B-Instruct or mistralai/Mistral-7B-Instruct-v0.2
DATASET_NAME="country_cities"       # country_cities, artist_songs, actor_movies
TEMPLATE_IDX=1   # 1, 2, 3
CACHE_DIR=""

CUDA_VISIBLE_DEVICES=0 python src/causal_tracing/main.py \
    --project_root_dir "${PROJECT_ROOT_DIR}" \
    --examine_type "${EXAMINE_TYPE}" \
    --model_name "${MODEL_NAME}" \
    --dataset_name "${DATASET_NAME}" \
    --template_idx "${TEMPLATE_IDX}" \
    --cache_dir "${CACHE_DIR}" \

#bash scripts/run_causal_tracing.sh
