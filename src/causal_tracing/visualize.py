import argparse
import json
import os

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer

from src.analysis_module.common_utils import get_model_name_for_visualization, reformat_var_name_for_visualization, \
    TEMPLATE_NUM
from src.causal_tracing.utils import merge_causal_tracing_figures, find_token_range, plot_trace_heatmap

ALL_KINDS = ["attn", "mlp"]
KIND_TO_FULL_NAME = {"attn": "Attn", "mlp": "MLP"}
NOISE_TARGET_TO_FULL_NAME = {"subject": "Subject", "prev_ans_1": r"$o^{(1)}$", "prev_ans_2": r"$o^{(2)}$"}
ANSWER_STEP_STR_TO_OBJECT_PROB_DIFF_LATEX = {"1": r"$\bigtriangleup~p(o^{(1)})$", "2": r"$\bigtriangleup~p(o^{(2)})$", "3": r"$\bigtriangleup~p(o^{(3)})$"}

TASK_TO_RELATION_NAME = {
    "country_cities": "cities",
    "artist_songs": "songs",
    "actor_movies": "movies",
}

ALL_DATASET_NAMES = ["country_cities", "artist_songs", "actor_movies"]

ALL_MODEL_NAMES = ["meta-llama/Meta-Llama-3-8B-Instruct", "mistralai/Mistral-7B-Instruct-v0.2"]

TOKENIZER_CONFIGS = {
    "add_bos_token": False,
    "add_eos_token": False
}

MODEL_NAME_TO_TOKENIZER = {
    "meta-llama/Meta-Llama-3-8B-Instruct": AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct", **TOKENIZER_CONFIGS),
    "mistralai/Mistral-7B-Instruct-v0.2": AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2", **TOKENIZER_CONFIGS),
}

ANSWER_STEP_TO_POSITION_TOKEN_NAMES = {
    "step_1": ["before relation tokens", "relation", "further tokens", "first subject token", "middle subject tokens", "last subject token", "further tokens", "1", "further tokens", r"last token before $o^{(1)}$"],
    "step_2": ["before relation tokens", "relation", "further tokens", "first subject token", "middle subject tokens", "last subject token", "further tokens", "1", "further tokens", r"last token before $o^{(1)}$", r"$o^{(1)}$ 1st token", r"$o^{(1)}$ middle tokens", r"$o^{(1)}$ last tokens", "further tokens", "2", "further tokens", r"last token before $o^{(2)}$"],
    "step_3": ["before relation tokens", "relation", "further tokens", "first subject token", "middle subject tokens", "last subject token", "further tokens", "1", "further tokens", r"last token before $o^{(1)}$", r"$o^{(1)}$ 1st token", r"$o^{(1)}$ middle tokens", r"$o^{(1)}$ last tokens", "further tokens", "2", "further tokens", r"last token before $o^{(2)}$", r"$o^{(2)}$ 1st token", r"$o^{(2)}$ middle tokens", r"$o^{(2)}$ last tokens", "further tokens", "3", "further tokens", r"last token before $o^{(3)}$"],
}
# Corresponds to the array above (not individual instance)
NOISE_TARGET_TOKEN_TO_RANGE = {
    "subject": [3, 6],
    "prev_ans_1": [9, 12],
    "prev_ans_2": [15, 18],
}

sub_result_dir_step_target_token = [
    ("causal_tracing_noise_subject/noise_subject_step_1", 1, "subject"),
    ("causal_tracing_noise_subject/noise_subject_step_2", 2, "subject"),
    ("causal_tracing_noise_subject/noise_subject_step_3", 3, "subject"),
    ("causal_tracing_noise_prev_ans/pred_2_noise_1", 2, "prev_ans_1"),
    ("causal_tracing_noise_prev_ans/pred_3_noise_1", 3, "prev_ans_1"),
    ("causal_tracing_noise_prev_ans/pred_3_noise_2", 3, "prev_ans_2"),
]


def get_position_token_vals(input_line_dict, tokenizer):
    input_ids = input_line_dict["input_ids"]
    scores = torch.tensor(np.array(input_line_dict["scores"]))
    position_token_vals = []

    def mean_score(start_idx, end_idx):
        # zero scores for invalid list
        return np.zeros(scores.size(1)).tolist() if start_idx >= end_idx else scores[start_idx:end_idx, :].mean(dim=0).tolist()

    relation_start_end_idx = input_line_dict["relation_start_end_idx"]
    subject_start_end_idx = input_line_dict["subject_start_end_idx"]
    subject_token_num = subject_start_end_idx[1] - subject_start_end_idx[0]
    one_dot_start_end_idx = find_token_range(tokenizer, input_ids, "1.")
    assert one_dot_start_end_idx[0] != -1
    first_answer_start_end_idx = input_line_dict["three_answers_start_end_idx"][0]

    position_token_vals.extend([
        mean_score(0, relation_start_end_idx[0]),  # Before relation
        mean_score(relation_start_end_idx[0], relation_start_end_idx[0] + 1),  # Relation tokens (cities, songs, movies)
        mean_score(relation_start_end_idx[0] + 1, subject_start_end_idx[0]),  # Further tokens
        mean_score(subject_start_end_idx[0], subject_start_end_idx[0] + 1) if subject_token_num > 1 else np.zeros(scores.size(1)).tolist(),     # First subject token
        mean_score(subject_start_end_idx[0] + 1, subject_start_end_idx[1] - 1) if subject_token_num > 2 else np.zeros(scores.size(1)).tolist(),     # Middle subject tokens
        mean_score(subject_start_end_idx[1] - 1, subject_start_end_idx[1]), # Last subject token
        mean_score(subject_start_end_idx[1], one_dot_start_end_idx[0]),     # Further tokens
        mean_score(one_dot_start_end_idx[0], one_dot_start_end_idx[0] + 1),     # 1
        mean_score(one_dot_start_end_idx[0] + 1, first_answer_start_end_idx[0] - 1),     # Further tokens
        mean_score(first_answer_start_end_idx[0] - 1, first_answer_start_end_idx[0]),     # Last token before answer 1
    ])
    
    def helper(curr_ans_idx):
        curr_ans_start_end_idx = input_line_dict["three_answers_start_end_idx"][curr_ans_idx-1]
        curr_ans_token_num = curr_ans_start_end_idx[1] - curr_ans_start_end_idx[0]
        next_ans_dot_start_end_idx = find_token_range(tokenizer, input_ids, f"{curr_ans_idx + 1}.")
        next_ans_start_end_idx = input_line_dict["three_answers_start_end_idx"][curr_ans_idx]
        position_token_vals.extend([
            mean_score(curr_ans_start_end_idx[0], curr_ans_start_end_idx[0] + 1) if curr_ans_token_num > 1 else np.zeros(scores.size(1)).tolist(),     # 1st answer 1st token
            mean_score(curr_ans_start_end_idx[0] + 1, curr_ans_start_end_idx[1] - 1) if curr_ans_token_num > 2 else np.zeros(scores.size(1)).tolist(),     # 1st answer middle tokens
            mean_score(curr_ans_start_end_idx[1] - 1, curr_ans_start_end_idx[1]),     # current answer last tokens
            mean_score(curr_ans_start_end_idx[1], next_ans_dot_start_end_idx[0]),     # Further tokens
            mean_score(next_ans_dot_start_end_idx[0], next_ans_dot_start_end_idx[0] + 1),     # next answer index
            mean_score(next_ans_dot_start_end_idx[0] + 1, next_ans_start_end_idx[0] - 1),  # Further tokens
            mean_score(next_ans_start_end_idx[0] - 1, next_ans_start_end_idx[0]), # Last token before the next answer
        ])


    if input_line_dict["target_answer_idx"] >= 2:
        helper(1)
        if input_line_dict["target_answer_idx"] == 3:
            helper(2)

    return np.array(position_token_vals)



def calculate_avg_causal_tracing_results_helper(input_result_jsonl, case_str, noise_target_str, kind, window=10, model_name=None, calculate_avg=True):
    """
    @param input_result_jsonl: str
    @param case_str: step_1, step_2, OR step_3
    @param noise_target_str: subject, prev_ans_1, OR prev_ans_2
    @param kind: attn, mlp, OR None (hidden_state)
    @param window: int
    @param model_name: str
    @param calculate_avg: bool (False when summing up scores to compute the micro average for the final macro average)
    """
    with open(input_result_jsonl, "r") as input_f:
        input_result_lines = input_f.readlines()
    scores = None
    all_low_scores = []
    all_high_scores = []
    for idx, pred_line in enumerate(input_result_lines):
        pred_line = json.loads(pred_line)
        all_low_scores.append(pred_line["low_score"])
        all_high_scores.append(pred_line["high_score"])
        pred_line["scores"] = torch.tensor(pred_line["scores"])
        tokenizer = MODEL_NAME_TO_TOKENIZER[model_name if model_name is not None else pred_line["model"]]
        position_token_vals = get_position_token_vals(pred_line, tokenizer)
        if scores is None:
            scores = position_token_vals
        else:
            scores += position_token_vals
    answer_step_str = case_str.split("_")[1]
    if calculate_avg:
        scores /= len(input_result_lines)
        low_score = np.average(all_low_scores).item()
        high_score = np.average(all_high_scores).item()
    else:
        low_score = np.sum(all_low_scores)
        high_score = np.sum(all_high_scores)
    results = {
        "scores": torch.tensor(scores),
        "low_score": low_score,
        "high_score": high_score,
        "input_tokens": ANSWER_STEP_TO_POSITION_TOKEN_NAMES[case_str],
        "subject_range": NOISE_TARGET_TOKEN_TO_RANGE[noise_target_str],
        "answer": ANSWER_STEP_STR_TO_OBJECT_PROB_DIFF_LATEX[answer_step_str],
        "window": window,
        "kind": kind,
        "line_cnt": len(input_result_lines),
    }

    return results


def visualize_dataset_model_specific_result(result_dir, model_name, dataset_name, template_idx):
    result_dir = f"{result_dir}/{dataset_name}/{model_name}/prompt_template_{template_idx}"
    all_fn_list = [[], []]  # Subject figures in the first list, prev_ans figures in the second list (will be merged together into one figure)
    progress = tqdm(total=len(sub_result_dir_step_target_token) * len(ALL_KINDS))
    for idx, (tmp_dir, pred_step, noise_token_str) in enumerate(sub_result_dir_step_target_token):
        for kind in ALL_KINDS:
            instance_name = f"{result_dir}/{tmp_dir}/{kind}_causal_tracing"
            has_saved = os.path.exists(f"{instance_name}.png")
            if not has_saved:
                fn = f"{result_dir}/{tmp_dir}/{kind}_result.jsonl"
                results = calculate_avg_causal_tracing_results_helper(
                    fn,
                    f"step_{pred_step}",
                    noise_token_str,
                    kind,
                    model_name=model_name
                )
                # Plot individual tracing figure
                plot_trace_heatmap(
                    results,
                    instance_name=instance_name,
                    title=f"Impact of Restoring {KIND_TO_FULL_NAME[kind]} when Noising {NOISE_TARGET_TO_FULL_NAME[noise_token_str]} at {f'step_{pred_step}'.replace('_', ' ').capitalize()} on {dataset_name.capitalize()}",
                    visualize=True
                )
            all_fn_list[int(idx / 3)].append(f"{instance_name}.png")  # Subject figures in the first list, prev_ans figures in the second list
            progress.update(1)

    # Merge figures: six for the subject and six for prev ans for specific model and dataset
    model_name_for_visualize = get_model_name_for_visualization(model_name)
    dataset_name_for_visualize = reformat_var_name_for_visualization(dataset_name)
    # Merge subject figures
    merge_causal_tracing_figures(all_fn_list[0],
                                 f"Noising Subject Causal Tracing: {model_name_for_visualize} on {dataset_name_for_visualize} (Template {template_idx})",
                                 f"{result_dir}/causal_tracing_figures/noise_subject_causal_tracing.png")
    # Merge prev ans figures
    merge_causal_tracing_figures(all_fn_list[1],
                                 f"Noising Previous Answer Causal Tracing: {model_name_for_visualize} on {dataset_name_for_visualize} (Template {template_idx})",
                                 f"{result_dir}/causal_tracing_figures/noise_prev_ans_causal_tracing.png")


def visualize_macro_avg(input_dir):
    output_dir = os.path.join(input_dir, "causal_tracing_avg_figures")

    progress = tqdm(total=len(sub_result_dir_step_target_token) * len(ALL_KINDS) * len(ALL_DATASET_NAMES) * len(ALL_MODEL_NAMES) * TEMPLATE_NUM)
    for idx, (tmp_dir, pred_step, noise_token_str) in enumerate(sub_result_dir_step_target_token):
        for kind in ALL_KINDS:
            # Calculate macro avg results across three datasets
            dataset_name_to_results = {dataset_name: {} for dataset_name in ALL_DATASET_NAMES}
            # Avg within one dataset
            for dataset_name in ALL_DATASET_NAMES:
                tmp_dataset_total_result = {}
                for tmp_model_name in ALL_MODEL_NAMES:
                    tmp_data_model_results = {}
                    for template_idx in range(1, TEMPLATE_NUM + 1):
                        fn = f"{input_dir}/{dataset_name}/{tmp_model_name}/prompt_template_{template_idx}/{tmp_dir}/{kind}_result.jsonl"

                        result = calculate_avg_causal_tracing_results_helper(
                            fn,
                            f"step_{pred_step}",
                            noise_token_str,
                            kind,
                            model_name=tmp_model_name,
                            calculate_avg=True,
                        )
                        if tmp_data_model_results == {}:
                            tmp_data_model_results = result
                        else:
                            tmp_data_model_results["scores"] += result["scores"]
                            tmp_data_model_results["low_score"] += result["low_score"]
                            tmp_data_model_results["high_score"] += result["high_score"]

                    progress.update(1)

                    # For each model: Average across templates
                    if tmp_dataset_total_result == {}:
                        tmp_dataset_total_result = {"scores": 0, "low_score": 0, "high_score": 0}
                    tmp_dataset_total_result["scores"] += tmp_data_model_results["scores"] / TEMPLATE_NUM
                    tmp_dataset_total_result["low_score"] += tmp_data_model_results["low_score"] / TEMPLATE_NUM
                    tmp_dataset_total_result["high_score"] += tmp_data_model_results["high_score"] / TEMPLATE_NUM

                # Average across models
                tmp_dataset_total_result["scores"] /= len(ALL_MODEL_NAMES)
                tmp_dataset_total_result["low_score"] /= len(ALL_MODEL_NAMES)
                tmp_dataset_total_result["high_score"] /= len(ALL_MODEL_NAMES)
                dataset_name_to_results[dataset_name] = tmp_dataset_total_result

            results = {
                "scores": (torch.stack([dataset_name_to_results[dataset_name]["scores"] for dataset_name in ALL_DATASET_NAMES]).sum(dim=0) / len(ALL_DATASET_NAMES)).tolist(),
                "low_score": np.average([dataset_name_to_results[dataset_name]["low_score"] for dataset_name in ALL_DATASET_NAMES]).item(),
                "high_score": np.average([dataset_name_to_results[dataset_name]["high_score"] for dataset_name in ALL_DATASET_NAMES]).item(),
                "input_tokens": ANSWER_STEP_TO_POSITION_TOKEN_NAMES[f"step_{pred_step}"],
                "subject_range": NOISE_TARGET_TOKEN_TO_RANGE[noise_token_str],
                "answer": ANSWER_STEP_STR_TO_OBJECT_PROB_DIFF_LATEX[f"{pred_step}"],
                "window": 10,   # Default val
                "kind": kind,
            }

            instance_name = f"{output_dir}/{tmp_dir}/{kind}_causal_tracing"
            print(f"Plot {instance_name}.png")
            # Plot individual tracing figure
            plot_trace_heatmap(
                results,
                instance_name=instance_name,
                title=f"Impact of Restoring {KIND_TO_FULL_NAME[kind]} when Noising {NOISE_TARGET_TO_FULL_NAME[noise_token_str]} at {f'step_{pred_step}'.replace('_', ' ').capitalize()}",
                visualize=True,
            )


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--project_root_dir", type=str, required=True)
    arg_parser.add_argument("--dataset_name", type=str, required=False)
    arg_parser.add_argument("--model_name", type=str, required=False)
    arg_parser.add_argument("--template_idx", type=str, required=False)

    args = arg_parser.parse_args()
    # When visualizing macro average results across all models, datasets, and prompt templates, all or none of dataset_name, model_name, template_idx should be "macro_avg"
    # Otherwise, choose specific dataset_name, model_name, and template_idx
    assert (args.dataset_name == "macro_avg") == (args.model_name == "macro_avg") == (args.template_idx == "macro_avg")
    if args.dataset_name == "macro_avg":
        print(f"Visualize macro_avg result")
        visualize_macro_avg(os.path.join(args.project_root_dir, "datasets"))
    else:
        print(f"Visualize result of {args.model_name} on {args.dataset_name}")
        visualize_dataset_model_specific_result(
            os.path.join(args.project_root_dir, "datasets"),
            args.model_name,
            args.dataset_name,
            args.template_idx
        )

    # bash scripts/visualize_causal_tracing.sh