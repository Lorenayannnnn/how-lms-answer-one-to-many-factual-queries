
export PYTHONPATH=:${PYTHONPATH}

result_dir="outputs/decode_attn_mlp_outputs"   # Set result_dir that contains outputs of specific models and datasets (e.g. outputs/decode_attention_mlp_outputs)
# Set both dataset_name and model_name to be "macro_avg" if you want to visualize the macro-average results across all datasets and models
dataset_name="macro_avg"    # "country_cities", "artist_songs", "actor_movies", "macro_avg"
model_name="macro_avg"    # "meta-llama/Meta-Llama-3-8B-Instruct", "mistralai/Mistral-7B-Instruct-v0.2", "macro_avg"
omit_early_layers=True    # "True" for omitting the first 15 layers (figures in the paper)

python src/analysis_module/visualize_decode_attention_mlp_outputs.py \
    --result_dir ${result_dir} \
    --dataset_name ${dataset_name} \
    --model_name ${model_name} \
    --omit_early_layers ${omit_early_layers}

#bash scripts/visualize_decode_attention_mlp_output.sh
