export PYTHONPATH=:${PYTHONPATH}

result_dir=./outputs/analyze_critical_tokens    # dir that contains results of all models and datasets

# Set specific model, dataset, and prompt template.
# If you have all the results, set everything to macro_avg to visualize macro avg results across all models, datasets, and prompt templates.
dataset_name="country_cities"    # "country_cities", "artist_songs", "actor_movies", "macro_avg"
model_name="meta-llama/Meta-Llama-3-8B-Instruct"    # "meta-llama/Meta-Llama-3-8B-Instruct", "mistralai/Mistral-7B-Instruct-v0.2", "macro_avg"
template_idx="1"   # "1", "2", "3", "macro_avg"
omit_early_layers=True   # "True" for omitting the first 15 layers (figures in the paper). else "False"

python src/analysis_module/visualize_critical_token_outputs.py \
    --result_dir ${result_dir} \
    --dataset_name ${dataset_name} \
    --model_name ${model_name} \
    --template_idx ${template_idx} \
    --omit_early_layers ${omit_early_layers}

#Loop through specific datasets and model_names
#for dataset_name in "country_cities" "artist_songs" "actor_movies"
#do
#    for model_name in "meta-llama/Meta-Llama-3-8B-Instruct" "mistralai/Mistral-7B-Instruct-v0.2"
#    do
#        python src/analysis_module/visualize_critical_token_outputs.py \
#            --result_dir ${result_dir} \
#            --dataset_name ${dataset_name} \
#            --model_name ${model_name} \
#            --omit_early_layers ${omit_early_layers}
#    done
#done

#bash scripts/visualize_critical_token_outputs.sh
