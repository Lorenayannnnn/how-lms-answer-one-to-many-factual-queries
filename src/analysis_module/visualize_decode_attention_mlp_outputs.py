import argparse
import json
import os

import matplotlib
import numpy as np
from matplotlib import pyplot as plt

from src.analysis_module.common_utils import get_component_full_name, COLOR_NAME_TO_RGB, merge_figures, \
    reformat_var_name_for_visualization, get_model_name_for_visualization, \
    merge_dataset_model_specific_token_level_figures, ALL_TOKENS_TO_BE_VISUALIZED
from src.common_utils import ALL_DATASET_NAMES, ALL_MODEL_NAMES


def visualize_token_lens_three_ans_to_layer_result(layer_num, three_ans_idx_to_layer_result, component_name, target_ans_idx, output_dir, input_omit_early_layers):
    """
    @param layer_num: total number of layers the model has
    @param three_ans_idx_to_layer_result: [[(#layer_num element)], [(#layer_num element)], [(#layer_num element)]]
    @param component_name: "attn", "mlp", or "hidden_state"
    @param target_ans_idx: 1, 2, or 3 (predict step)
    @param output_dir: directory to save the plot
    @param input_omit_early_layers: whether to omit early layers
    """
    os.makedirs(output_dir, exist_ok=True)
    output_figure_fn = f"{output_dir}/{component_name}_avg_logit_at_step_#{target_ans_idx}.png"
    component_name = get_component_full_name(component_name)
    layer_names = [f"{layer_idx}" for layer_idx in range(1, layer_num + 1)]

    answer_idx_to_line_style = ['-'] * len(ALL_TOKENS_TO_BE_VISUALIZED)

    plt.figure(figsize=(10, 7))
    plt.rcParams.update({'font.size': 20})
    plt.rcParams['lines.linewidth'] = 3
    plt.rcParams['lines.markersize'] = 9

    if input_omit_early_layers:
        layer_names = layer_names[OMIT_LAYER_START_IDX:]
        for k, v in three_ans_idx_to_layer_result.items():
            three_ans_idx_to_layer_result[k] = v[OMIT_LAYER_START_IDX:]
        plt.figure(figsize=(6, 6))

    plt.plot(layer_names, three_ans_idx_to_layer_result["answer_1"], marker='o', label='Answer 1', linestyle=answer_idx_to_line_style[0], color=COLOR_NAME_TO_RGB['blue'])
    plt.plot(layer_names, three_ans_idx_to_layer_result["answer_2"], marker='s', label='Answer 2', linestyle=answer_idx_to_line_style[1], color=COLOR_NAME_TO_RGB['orange'])
    plt.plot(layer_names, three_ans_idx_to_layer_result["answer_3"], marker='^', label='Answer 3', linestyle=answer_idx_to_line_style[2], color=COLOR_NAME_TO_RGB['green'])
    plt.plot(layer_names, three_ans_idx_to_layer_result["subject"], marker='*', label='Subject', linestyle=answer_idx_to_line_style[-1], color=COLOR_NAME_TO_RGB['red'])

    if input_omit_early_layers and "macro_avg" in output_dir:
        if component_name == "Attention":
            plt.ylim(top=1.3, bottom=-0.9)
        elif component_name == "MLP":
            plt.ylim(top=6.5, bottom=-0.5)

    plt.axhline(y=0, color='r', linestyle='--')
    plt.gca().yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%.1f'))

    plt.xlabel(f'Layer')
    plt.ylabel("Logit")
    plt.title(f"Step #{target_ans_idx}")

    plt.xticks([layer_name if idx % 2 == 0 else "" for idx, layer_name in enumerate(layer_names)])
    plt.tight_layout()
    plt.savefig(output_figure_fn)

    plt.clf()
    matplotlib.pyplot.close()

    return output_figure_fn


def process_and_avg_result_helper(input_json_fn, calculate_avg=True):
    token_name_to_layer_idx_vals = {token_name: [[] for _ in range(LAYER_CNT)] for token_name in ALL_TOKENS_TO_BE_VISUALIZED}
    entry_cnt = 0
    with open(input_json_fn, "r") as f:
        for line in f:
            line = json.loads(line)
            token_name_to_layer_idx_vals["answer_1"][line["layer_idx"]].append(line[f"three_answer_first_token_logit"][0])
            token_name_to_layer_idx_vals["answer_2"][line["layer_idx"]].append(line[f"three_answer_first_token_logit"][1])
            token_name_to_layer_idx_vals["answer_3"][line["layer_idx"]].append(line[f"three_answer_first_token_logit"][2])
            token_name_to_layer_idx_vals["subject"][line["layer_idx"]].append(line[f"subject_first_token_logit"])
            if line["layer_idx"] == 0:      # Only count once for each entry
                entry_cnt += 1
    token_name_to_layer_idx_final_val = {token_name: [0] * LAYER_CNT for token_name in ALL_TOKENS_TO_BE_VISUALIZED}
    for token_name, layer_idx_vals in token_name_to_layer_idx_vals.items():
        for layer_idx, vals in enumerate(layer_idx_vals):
            token_name_to_layer_idx_final_val[token_name][layer_idx] = sum(vals)
            if calculate_avg:
                token_name_to_layer_idx_final_val[token_name][layer_idx] /= len(vals)

    return token_name_to_layer_idx_final_val, entry_cnt


def visualize_dataset_model_specific_result(result_dir, dataset_name, model_name, omit_early_layers):
    data_model_specific_dir = f"{result_dir}/{args.dataset_name}/{args.model_name}"
    dataset_name_for_visualize = reformat_var_name_for_visualization(dataset_name)
    model_name_for_visualize = get_model_name_for_visualization(model_name)
    component_name_to_step_fn_list = {}
    for component_name in ALL_COMPONENTS:
        tmp_output_dir = f"{data_model_specific_dir}/{'full_figures' if not omit_early_layers else 'omit_early_layer_figures'}/{component_name}"
        output_figure_fn_list = []
        for target_answer_idx in [1, 2, 3]:
            tmp_jsonl_fn = f"{data_model_specific_dir}/{target_answer_idx}/{component_name}_output.jsonl"
            token_name_to_layer_idx_avg_val, _ = process_and_avg_result_helper(tmp_jsonl_fn)

            output_figure_fn = visualize_token_lens_three_ans_to_layer_result(LAYER_CNT,
                                                                              token_name_to_layer_idx_avg_val,
                                                                              component_name, target_answer_idx,
                                                                              tmp_output_dir,
                                                                              omit_early_layers)
            output_figure_fn_list.append(output_figure_fn)
            component_name_to_step_fn_list.setdefault(component_name, []).append(output_figure_fn)
        # Merge three images
        if not omit_early_layers:
            title = f"{get_component_full_name(component_name)} Output Logit: {model_name_for_visualize} on {dataset_name_for_visualize}"
            merge_figures(
                output_figure_fn_list, title,
                f"{data_model_specific_dir}/full_figures/{component_name}_logit.png",
                "decode_attn_mlp"
            )

    if omit_early_layers:     # Six images in a row: left three for attention, right three for MLP
        tmp_fn_list = [fn for fn in component_name_to_step_fn_list['attn']] + [fn for fn in component_name_to_step_fn_list['mlp']]
        merge_dataset_model_specific_token_level_figures(
            tmp_fn_list,
            f"{model_name_for_visualize} on {dataset_name_for_visualize}",
            f"Attention Output Logits",
            f"MLP Output Logits",
            f"{data_model_specific_dir}/attn_mlp_output_logit_omit_early_layers.png",
        )


def visualize_macro_avg(result_dir, omit_early_layers):
    output_dir = os.path.join(result_dir, "macro_avg_figures")

    component_name_to_step_fn_list = {}
    for component_name in ALL_COMPONENTS:
        tmp_output_dir = f"{output_dir}/{'full_figures' if not omit_early_layers else 'omit_early_layer_figures'}/{component_name}"
        for target_answer_idx in [1, 2, 3]:
            output_figure_fn_list = []
            # Average across models for each dataset
            dataset_name_to_token_name_to_layer_idx_vals = {dataset_name: {} for dataset_name in ALL_DATASET_NAMES}
            for dataset_name in ALL_DATASET_NAMES:
                tmp_dataset_results = {}
                tmp_dataset_line_cnt = 0
                for model_name in ALL_MODEL_NAMES:
                    tmp_jsonl_fn = f"{result_dir}/{dataset_name}/{model_name}/{target_answer_idx}/{component_name}_output.jsonl"
                    tmp_result, tmp_line_cnt = process_and_avg_result_helper(tmp_jsonl_fn, calculate_avg=False)
                    if tmp_dataset_results == {}:
                        tmp_dataset_results = tmp_result
                    else:
                        # Add up for each layer
                        for token_name, layer_vals in tmp_result.items():
                            tmp_dataset_results[token_name] = np.sum([np.array(tmp_dataset_results[token_name]), np.array(layer_vals)], axis=0)
                    tmp_dataset_line_cnt += tmp_line_cnt
                # Average for each dataset
                for token_name in ALL_TOKENS_TO_BE_VISUALIZED:
                    tmp_dataset_results[token_name] = (tmp_dataset_results[token_name] / tmp_dataset_line_cnt).tolist()
                dataset_name_to_token_name_to_layer_idx_vals[dataset_name] = tmp_dataset_results

            # Macro avg across datasets
            token_name_to_layer_idx_avg_val = {token_name: [0] * LAYER_CNT for token_name in ALL_TOKENS_TO_BE_VISUALIZED}
            for token_name in ALL_TOKENS_TO_BE_VISUALIZED:
                for layer_idx in range(LAYER_CNT):
                    token_name_to_layer_idx_avg_val[token_name][layer_idx] = sum([dataset_name_to_token_name_to_layer_idx_vals[dataset_name][token_name][layer_idx] for dataset_name in ALL_DATASET_NAMES]) / len(ALL_DATASET_NAMES)

            output_figure_fn = visualize_token_lens_three_ans_to_layer_result(LAYER_CNT,
                                                                              token_name_to_layer_idx_avg_val,
                                                                              component_name, target_answer_idx,
                                                                              tmp_output_dir,
                                                                              omit_early_layers)
            output_figure_fn_list.append(output_figure_fn)
            component_name_to_step_fn_list.setdefault(component_name, []).append(output_figure_fn)


if __name__ == '__main__':
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument("--result_dir", type=str, required=True)
    args_parser.add_argument("--dataset_name", type=str, required=True)      # ["country_cities", "artist_songs", "actor_movies", "macro_avg"]
    args_parser.add_argument("--model_name", type=str, required=True)        # ["meta-llama/Meta-Llama-3-8B-Instruct", "mistralai/Mistral-7B-Instruct-v0.2", "macro_avg"]
    args_parser.add_argument("--omit_early_layers", type=bool, required=True)        # True for reproducing figure with 6 subfigures in a row in the paper. False for full figures in the appendix
    args = args_parser.parse_args()

    LAYER_CNT = 32
    OMIT_LAYER_START_IDX = 15
    ALL_COMPONENTS = ["attn", "mlp"]
    # When visualizing macro average results across all models and datasets, both or neither of dataset_name and model_name should be "macro_avg"
    # Otherwise, choose specific dataset_name and model_name
    assert (args.dataset_name == "macro_avg") == (args.model_name == "macro_avg")
    if args.dataset_name == "macro_avg":
        visualize_macro_avg(args.result_dir, omit_early_layers=args.omit_early_layers)
        print(f"Visualize results in {args.result_dir}")
    else:
        visualize_dataset_model_specific_result(
            result_dir=args.result_dir,
            dataset_name=args.dataset_name,
            model_name=args.model_name,
            omit_early_layers=args.omit_early_layers
        )
        print(f"Visualize results in {args.result_dir}/{args.dataset_name}/{args.model_name}")


#bash scripts/visualize_decode_attention_mlp_output.sh
