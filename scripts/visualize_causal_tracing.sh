
export PYTHONPATH=:${PYTHONPATH}

PROJECT_ROOT_DIR=.
# Set specific model, dataset, and prompt template.
# If you have all the results, set everything to macro_avg to visualize macro avg results across all models, datasets, and prompt templates.
MODEL_NAME="meta-llama/Meta-Llama-3-8B-Instruct"  # meta-llama/Meta-Llama-3-8B-Instruct, mistralai/Mistral-7B-Instruct-v0.2, macro_avg
DATASET_NAME="country_cities"       # country_cities, artist_songs, actor_movies, macro_avg
TEMPLATE_IDX="1"   # 1, 2, 3, macro_avg


python src/causal_tracing/visualize.py \
    --project_root_dir "${PROJECT_ROOT_DIR}" \
    --model_name "${MODEL_NAME}" \
    --dataset_name "${DATASET_NAME}" \
    --template_idx "${TEMPLATE_IDX}"

#bash scripts/visualize_causal_tracing.sh
