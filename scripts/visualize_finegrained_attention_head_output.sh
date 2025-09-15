export PYTHONPATH=:${PYTHONPATH}

PROJECT_ROOT_DIR=.
RESULT_DIR=./outputs/examine_finegrained_attention_head

# TODO: current code only supports macro-averaging across models and datasets on one prompt template (will update this later when have time)
template_idx=1

python src/analysis_module/visualize_finegraind_attn_head_outputs.py \
    --project_root_dir ${PROJECT_ROOT_DIR} \
    --result_dir ${RESULT_DIR} \
    --template_idx ${template_idx}

#bash scripts/visualize_finegrained_attention_head_output.sh
