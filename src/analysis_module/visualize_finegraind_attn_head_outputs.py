import argparse
import json
import os

import matplotlib
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

from src.analysis_module.common_utils import merge_figures
from src.common_utils import ALL_DATASET_NAMES, ALL_MODEL_NAMES

BEHAVIOR_TO_COLOR = {
    "promotion": "green",
    "suppression": "red",
    "nothing": "grey",
    "both": "brown"
}


def get_logit_behavior(logit, mean, std, k=1):
    if logit > mean + k * std:
        return "promotion"
    elif logit < mean - k * std:
        return "suppression"
    else:
        return "nothing"


def get_head_behavior(input_entry, mean, std):
    subject_behavior = get_logit_behavior(input_entry["subject_first_token_logit"], mean, std)
    answer_1_behavior = get_logit_behavior(input_entry["three_answer_first_token_logit"][0], mean, std)
    answer_2_behavior = get_logit_behavior(input_entry["three_answer_first_token_logit"][1], mean, std)
    answer_3_behavior = get_logit_behavior(input_entry["three_answer_first_token_logit"][2], mean, std)

    all_behaviors = [subject_behavior, answer_1_behavior, answer_2_behavior, answer_3_behavior]
    result_dict = {"promotion": 0, "suppression": 0, "nothing": 0}
    if "promotion" in all_behaviors:
        result_dict["promotion"] += 1
    if "suppression" in all_behaviors:
        result_dict["suppression"] += 1
    if "promotion" not in all_behaviors and "suppression" not in all_behaviors:
        result_dict["nothing"] += 1

    return result_dict


def calculate_mean_std_of_each_layer():
    # dataset_to_model_to_answer_idx_to_layer_mean_std.json is already provided under the datasets directory.
    # You can rerun this to regenerate the results.
    dataset_to_model_to_answer_idx_to_layer_mean_std = {dataset_name: {
        model_name: {target_answer_idx: {layer_idx: {"mean": 0, "std": 0} for layer_idx in range(layer_cnt)} for
                     target_answer_idx in [1, 2, 3]} for model_name in ALL_MODEL_NAMES} for dataset_name in
                                                        ALL_DATASET_NAMES}
    for dataset_name in ALL_DATASET_NAMES:
        for model_name in ALL_MODEL_NAMES:
            for target_answer_idx in [1, 2, 3]:
                for layer_idx in tqdm(range(layer_cnt)):
                    tmp_jsonl_fn = f"{result_dir}/{dataset_name}/{model_name}/{target_answer_idx}/layer_{layer_idx}_output.jsonl"
                    tmp_all_logit_vals = []
                    with open(tmp_jsonl_fn, "r") as f:
                        for line in f:
                            line = json.loads(line)
                            for head_idx in range(head_num):
                                tmp_head_data = line[f"head_{head_idx}"]
                                tmp_logits = [tmp_head_data["subject_first_token_logit"], tmp_head_data["three_answer_first_token_logit"][0], tmp_head_data["three_answer_first_token_logit"][1], tmp_head_data["three_answer_first_token_logit"][2]]
                                tmp_all_logit_vals.extend(tmp_logits)

                    tmp_all_logit_vals = np.array(tmp_all_logit_vals)
                    tmp_mean = np.mean(tmp_all_logit_vals).item()
                    tmp_std = np.std(tmp_all_logit_vals).item()
                    dataset_to_model_to_answer_idx_to_layer_mean_std[dataset_name][model_name][target_answer_idx][layer_idx]["mean"] = tmp_mean
                    dataset_to_model_to_answer_idx_to_layer_mean_std[dataset_name][model_name][target_answer_idx][layer_idx]["std"] = tmp_std
    # Save the results
    with open(f"{project_root_dir}/datasets/dataset_to_model_to_answer_idx_to_layer_mean_std.json", "w") as f:
        f.write(json.dumps(dataset_to_model_to_answer_idx_to_layer_mean_std))


def visualize_head_promotion_suppression_rate(layer_head_idx_to_behavior_rate, output_fn, target_answer_idx=None):
    plt.clf()
    layer_head_idx_to_behavior_rate = {k: v for k, v in layer_head_idx_to_behavior_rate.items() if v["total"] > 0}
    layer_head_idx = sorted(layer_head_idx_to_behavior_rate.keys())  # Sorted layer indices
    promotion_rates = [layer_head_idx_to_behavior_rate[layer_head_idx]["promotion"] * 100 for layer_head_idx in layer_head_idx]
    suppression_rates = [layer_head_idx_to_behavior_rate[layer_head_idx]["suppression"] * 100 for layer_head_idx in layer_head_idx]

    plt.figure(figsize=(8, 8))
    plt.rcParams.update({'font.size': 20})
    plt.scatter(promotion_rates, suppression_rates, color="blue", alpha=0.5)
    plt.xlabel("Promotion Rate (%)")
    plt.ylabel("Suppression Rate (%)")
    title = f"Step {target_answer_idx}"
    plt.title(title)
    plt.tight_layout()

    plt.savefig(output_fn)
    matplotlib.pyplot.close()


def visualize_macro_avg_all_heads_promotion_suppression_rate():
    with open(f"{project_root_dir}/datasets/dataset_to_model_to_answer_idx_to_layer_mean_std.json", "r") as f:
        dataset_to_model_to_answer_idx_to_layer_mean_std = json.load(f)

    overall_output_dir = f"{result_dir}/figures/avg_figures/prompt_template_{template_idx}"     # TODO: current code only supports macro-averaging across models and datasets on one prompt template (will update this later when have time)
    os.makedirs(overall_output_dir, exist_ok=True)

    overall_target_idx_to_dataset_to_layer_head_idx_to_behavior_cnt = {target_idx: {
        dataset_name: {f"L{layer_idx}_H{head_idx}": {"promotion": 0, "suppression": 0, "total": 0} for head_idx in
                       range(head_num) for layer_idx in range(layer_cnt)} for dataset_name in ALL_DATASET_NAMES} for
                                                                       target_idx in [1, 2, 3]}
    for dataset_name in ALL_DATASET_NAMES:
        for model_name in ALL_MODEL_NAMES:
            for target_answer_idx in [1, 2, 3]:
                layer_head_idx_to_behavior_cnt = {f"L{layer_idx}_H{head_idx}": {"promotion": 0, "suppression": 0, "total": 0} for head_idx in range(head_num) for layer_idx in range(layer_cnt)}
                for layer_idx in tqdm(range(layer_cnt)):
                    tmp_layer_mean_std_dict = dataset_to_model_to_answer_idx_to_layer_mean_std[dataset_name][model_name][f'{target_answer_idx}'][f'{layer_idx}']
                    tmp_layer_mean, tmp_layer_std = tmp_layer_mean_std_dict["mean"], tmp_layer_mean_std_dict["std"]
                    tmp_jsonl_fn = f"{result_dir}/{dataset_name}/{model_name}/prompt_template_{template_idx}/{target_answer_idx}/layer_{layer_idx}_output.jsonl"
                    with open(tmp_jsonl_fn, "r") as f:
                        lines = f.readlines()
                        for line in lines:
                            line = json.loads(line)
                            for head_idx in range(head_num):
                                tmp_head_data = line[f"head_{head_idx}"]
                                tmp_head_behavior_dict = get_head_behavior(tmp_head_data, tmp_layer_mean, tmp_layer_std)
                                tmp_head_behavior_dict.pop("nothing")
                                for k, v in tmp_head_behavior_dict.items():
                                    layer_head_idx_to_behavior_cnt[f"L{layer_idx}_H{head_idx}"][k] += v
                                    overall_target_idx_to_dataset_to_layer_head_idx_to_behavior_cnt[target_answer_idx][dataset_name][f"L{layer_idx}_H{head_idx}"][k] += v
                                layer_head_idx_to_behavior_cnt[f"L{layer_idx}_H{head_idx}"]["total"] += 1
                                overall_target_idx_to_dataset_to_layer_head_idx_to_behavior_cnt[target_answer_idx][dataset_name][f"L{layer_idx}_H{head_idx}"]["total"] += 1

                avg_layer_head_idx_to_behavior_cnt = {k: {k2: v2 / layer_head_idx_to_behavior_cnt[k]["total"] for k2, v2 in v.items()} for k, v in layer_head_idx_to_behavior_cnt.items()}

                tmp_output_dir = f"{overall_output_dir}/{dataset_name}/{model_name}/{target_answer_idx}"
                tmp_output_fn = f"{tmp_output_dir}_head_promotion_suppression_rate.png"

                visualize_head_promotion_suppression_rate(avg_layer_head_idx_to_behavior_cnt, tmp_output_fn)

    # Avg within dataset first and then avg across datasets
    all_fn_list = []
    for target_idx, result_dict in overall_target_idx_to_dataset_to_layer_head_idx_to_behavior_cnt.items():
        dataset_name_to_avg_result = {dataset_name: {} for dataset_name in result_dict.keys()}
        for dataset_name, results in result_dict.items():
            overall_layer_head_idx_to_behavior_percentage = {k: {k2: v2 / results[k]["total"] for k2, v2 in v.items()} for k, v in results.items()}
            dataset_name_to_avg_result[dataset_name] = overall_layer_head_idx_to_behavior_percentage

        macro_avg_layer_head_idx_to_behavior = {k: {"promotion": 0, "suppression": 0, "total": 0} for k in dataset_name_to_avg_result["country_cities"].keys()}
        for layer_head_idx, _ in dataset_name_to_avg_result["country_cities"].items():
            for dataset_name, results in dataset_name_to_avg_result.items():
                macro_avg_layer_head_idx_to_behavior[layer_head_idx]["promotion"] += results[layer_head_idx]["promotion"]
                macro_avg_layer_head_idx_to_behavior[layer_head_idx]["suppression"] += results[layer_head_idx]["suppression"]
                macro_avg_layer_head_idx_to_behavior[layer_head_idx]["total"] += results[layer_head_idx]["total"]
        macro_avg_layer_head_idx_to_behavior_percentage = {k: {k2: v2 / macro_avg_layer_head_idx_to_behavior[k]["total"] for k2, v2 in v.items()} for k, v in macro_avg_layer_head_idx_to_behavior.items()}

        output_fn = f"{overall_output_dir}/head_promotion_suppression_rate{f'_step_{target_idx}' if target_idx is not None else ''}.png"
        all_fn_list.append(output_fn)
        visualize_head_promotion_suppression_rate(macro_avg_layer_head_idx_to_behavior_percentage, output_fn=output_fn, target_answer_idx=target_idx)
    # Create a single figure with all subplots
    merge_figures(all_fn_list, "Attention Head Promotion vs. Suppression Rate", f"{overall_output_dir}/head_promotion_vs_suppression_rate.png", exp_name="head_promotion_suppression_rate")

if __name__ == "__main__":
    layer_cnt = 32
    head_num = 32
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument("--project_root_dir", type=str, required=True)
    args_parser.add_argument("--result_dir", type=str, required=True)
    args_parser.add_argument("--template_idx", type=str, required=True)
    args = args_parser.parse_args()
    project_root_dir = args.project_root_dir
    result_dir = args.result_dir
    template_idx = args.template_idx

    visualize_macro_avg_all_heads_promotion_suppression_rate()

#bash scripts/visualize_finegrained_attention_head_output.sh
