
export PYTHONPATH=:${PYTHONPATH}

PROJECT_ROOT_DIR=.
# Set specific model and dataset.
# If you have all the results, set both to macro_avg to visualize macro avg results across all models and datasets
MODEL_NAME="macro_avg"  # meta-llama/Meta-Llama-3-8B-Instruct, mistralai/Mistral-7B-Instruct-v0.2, macro_avg
DATASET_NAME="macro_avg"       # country_cities, artist_songs, actor_movies, macro_avg

python src/causal_tracing/visualize.py \
    --project_root_dir "${PROJECT_ROOT_DIR}" \
    --model_name "${MODEL_NAME}" \
    --dataset_name "${DATASET_NAME}"

#bash scripts/visualize_causal_tracing.sh
