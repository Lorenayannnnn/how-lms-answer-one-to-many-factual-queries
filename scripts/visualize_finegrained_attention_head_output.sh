export PYTHONPATH=:${PYTHONPATH}

PROJECT_ROOT_DIR=.
RESULT_DIR=./examine_finegrained_attention_head
python src/analysis_module/visualize_finegraind_attn_head_outputs.py \
    --project_root_dir ${PROJECT_ROOT_DIR} \
    --result_dir ${RESULT_DIR} \

#bash scripts/visualize_finegrained_attention_head_output.sh
