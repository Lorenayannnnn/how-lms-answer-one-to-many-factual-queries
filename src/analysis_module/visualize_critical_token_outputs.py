import argparse
import json
import os.path

import matplotlib
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tqdm import tqdm

from src.analysis_module.common_utils import COLOR_NAME_TO_RGB, reformat_var_name_for_visualization, \
    get_model_name_for_visualization, merge_figures, merge_dataset_model_specific_token_level_figures, \
    ALL_TOKENS_TO_BE_VISUALIZED, TEMPLATE_NUM
from src.common_utils import STEP_TO_TOKEN_TYPE_NAME, ANS_TO_LATEX, ALL_DATASET_NAMES, ALL_MODEL_NAMES

EXP_TYPE_TO_COMPONENT_NAME = {
    "token_lens": "attn",
    "attn_knockout": "mlp"
}


def read_jsonl_to_csv(input_jsonl_fn):
    all_results = []
    with open(input_jsonl_fn, "r") as f:
        for line in f:
            all_results.append(json.loads(line))
    return pd.DataFrame(all_results)


def visualize_token_lens_three_ans_to_layer_result(token_name_to_layer_result, exp_type, token_type_name,
                                                   target_ans_idx, input_omit_early_layers, output_fn):
    """
    @param token_name_to_layer_result: {'answer_1': [#layer_num element], 'answer_2': [], 'answer_3': [], 'subject': []}
    @param exp_type: "token_lens" or "attn_knockout"
    @param token_type_name: "subject", "answer_1", "answer_2", or "last_token"
    @param target_ans_idx: 1, 2, or 3 (answer step)
    @param input_omit_early_layers: if True, omit the early layers (start from layer 15)
    @param output_fn: output file name
    """
    os.makedirs(os.path.dirname(output_fn), exist_ok=True)
    if exp_type == "token_lens":
        component_name = "Attention"
    elif exp_type == "attn_knockout":
        component_name = "MLP"
    else:
        raise ValueError(f"Visualization not implemented for exp_type {exp_type}")
    exp_type = reformat_var_name_for_visualization(exp_type)
    token_type_name = reformat_var_name_for_visualization(token_type_name)
    layer_names = [f"{layer_idx}" for layer_idx in range(1, LAYER_NUM + 1)]
    print(f"Visualized {exp_type} result of {token_type_name}")

    plt.figure(figsize=(10, 6))
    plt.rcParams.update({'font.size': 18})
    plt.rcParams['lines.linewidth'] = 3
    plt.rcParams['lines.markersize'] = 9

    if input_omit_early_layers:
        plt.figure(figsize=(6, 6))
        layer_names = layer_names[OMIT_LAYER_START_IDX:]
        for token_name, token_layer_results in token_name_to_layer_result.items():
            token_name_to_layer_result[token_name] = token_layer_results[OMIT_LAYER_START_IDX:]

    plt.axhline(y=0, color='r', linestyle='--')
    plt.plot(layer_names, token_name_to_layer_result['answer_1'], marker='o', label='Answer 1', linestyle='-', color=COLOR_NAME_TO_RGB['blue'])
    plt.plot(layer_names, token_name_to_layer_result['answer_2'], marker='s', label='Answer 2', linestyle='-', color=COLOR_NAME_TO_RGB['orange'])
    plt.plot(layer_names, token_name_to_layer_result['answer_3'], marker='^', label='Answer 3', linestyle='-', color=COLOR_NAME_TO_RGB['green'])
    plt.plot(layer_names, token_name_to_layer_result['subject'], marker='*', label='Subject', linestyle='-', color=COLOR_NAME_TO_RGB['red'])

    if input_omit_early_layers and "macro_avg_figures" in output_fn:
        if component_name == "Attention":
            if token_type_name == "Subject":
                plt.ylim(top=0.38, bottom=-0.2)
            elif token_type_name == "Answer 1" or token_type_name == "Answer 2":
                plt.ylim(top=0.1, bottom=-1.4)
            elif token_type_name == "Last Token":
                plt.ylim(top=1.0, bottom=-0.05)
        elif component_name == "MLP":
            if token_type_name == "Subject":
                plt.ylim(top=1.4, bottom=-1.1)
            elif token_type_name == "Answer 1":
                plt.ylim(top=0.6, bottom=-1.5)
            elif token_type_name == "Last Token":
                plt.ylim(top=0.2, bottom=-1.5)
    plt.tight_layout(pad=2)
    plt.xticks(range(len(layer_names)), [layer_name if idx % 2 == 0 else "" for idx, layer_name in enumerate(layer_names)])
    plt.gca().yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%.1f'))

    plt.xlabel(f'Layer')
    plt.ylabel(f"Logit")
    fig_title = f"Step #{target_ans_idx}"
    if "Answer" in token_type_name or "Ans" in token_type_name:
        if "Knockout" in exp_type:
            fig_title += ": Knockout " + ANS_TO_LATEX[token_type_name]
        else:
            fig_title += ": Attend to " + ANS_TO_LATEX[token_type_name]
    plt.title(fig_title)

    plt.savefig(output_fn)
    plt.clf()
    matplotlib.pyplot.close()


def process_dataset_model_specific_critical_tokens_result(input_result_dir: str, input_dataset_name, input_model_name, input_template_idx, input_omit_early_layers):
    """
    Visualize critical tokens' results of a model on the specified dataset
    """
    dataset_name_for_visualize = reformat_var_name_for_visualization(input_dataset_name)
    model_name_for_visualize = get_model_name_for_visualization(input_model_name)
    token_type_to_exp_type_to_step_fn_list = {}
    progress = tqdm(total=len(EXP_TYPE_TO_COMPONENT_NAME.keys()) * 3)
    for exp_type in EXP_TYPE_TO_COMPONENT_NAME.keys():
        component_name = EXP_TYPE_TO_COMPONENT_NAME[exp_type]
        tmp_output_dir = os.path.join(input_result_dir, "figures", "data_model_specific", f"{'full_figures' if not input_omit_early_layers else 'omit_early_layer_figures'}", input_dataset_name, input_model_name, f"prompt_template_{input_template_idx}")
        os.makedirs(tmp_output_dir, exist_ok=True)
        all_token_type_to_output_image_fn_list = {}
        for target_answer_idx in [1, 2, 3]:
            tmp_all_token_types = STEP_TO_TOKEN_TYPE_NAME[f"{target_answer_idx}"]
            token_type_fn_name_list = [f"{exp_type}_{tmp_token_type}_results.jsonl" for tmp_token_type in tmp_all_token_types]
            type_to_df = {type_name: read_jsonl_to_csv(f"{input_result_dir}/{input_dataset_name}/{input_model_name}/prompt_template_{input_template_idx}/{target_answer_idx}/{type_name}") for type_name in token_type_fn_name_list}

            for tmp_token_type_fn, tmp_token_type_name in zip(token_type_fn_name_list, tmp_all_token_types):
                token_name_to_layer_result = {token_name: [] for token_name in ALL_TOKENS_TO_BE_VISUALIZED}
                for layer_idx in range(LAYER_NUM):
                    if exp_type == "attn_knockout":
                        # Get results for the current layer
                        tmp_layer_three_answer_attn_knockout_results = np.array(type_to_df[tmp_token_type_fn][type_to_df[tmp_token_type_fn]["layer_idx"] == layer_idx][f"three_answer_first_token_attn_knockout_{component_name}_logit"].tolist()).transpose()       # shape: (3 answers, num_samples)
                        tmp_layer_three_answer_full_results = np.array(type_to_df[tmp_token_type_fn][type_to_df[tmp_token_type_fn]["layer_idx"] == layer_idx][f"three_answer_first_token_full_{component_name}_logit"].tolist()).transpose()       # shape: (3 answers, num_samples)
                        tmp_layer_subject_attn_knockout_results = np.array(type_to_df[tmp_token_type_fn][type_to_df[tmp_token_type_fn]["layer_idx"] == layer_idx][f"subject_first_token_attn_knockout_{component_name}_logit"].tolist())     # shape: (num_samples,)
                        tmp_layer_subject_full_results = np.array(type_to_df[tmp_token_type_fn][type_to_df[tmp_token_type_fn]["layer_idx"] == layer_idx][f"subject_first_token_full_{component_name}_logit"].tolist())     # shape: (num_samples,)

                        # Calculate average of full - knockout diff
                        for tmp_ans_idx in range(3):
                            token_name_to_layer_result[f"answer_{tmp_ans_idx+1}"].append(np.average(tmp_layer_three_answer_full_results[tmp_ans_idx] - tmp_layer_three_answer_attn_knockout_results[tmp_ans_idx], axis=0))
                        token_name_to_layer_result['subject'].append(np.average(tmp_layer_subject_full_results - tmp_layer_subject_attn_knockout_results, axis=0))
                    else:
                        # Token lens
                        tmp_layer_three_answer_results = np.array(type_to_df[tmp_token_type_fn][type_to_df[tmp_token_type_fn]["layer_idx"] == layer_idx][f"three_answer_first_token_logit"].tolist()).transpose()       # shape: (3 answers, num_samples)
                        tmp_layer_subject_results = np.array(type_to_df[tmp_token_type_fn][type_to_df[tmp_token_type_fn]["layer_idx"] == layer_idx][f"subject_first_token_logit"].tolist())     # shape: (num_samples,)

                        # Calculate average
                        for tmp_ans_idx in range(3):
                            token_name_to_layer_result[f"answer_{tmp_ans_idx+1}"].append(np.average(tmp_layer_three_answer_results[tmp_ans_idx], axis=0))
                        token_name_to_layer_result['subject'].append(np.average(tmp_layer_subject_results, axis=0))

                output_image_fn = f"{tmp_output_dir}/{tmp_token_type_name}/{exp_type}_at_step_#{target_answer_idx}.png"
                if tmp_token_type_name == "answer_1" or tmp_token_type_name == "answer_2":
                    tmp_token_image_fn_key = "prev_ans"
                else:
                    tmp_token_image_fn_key = tmp_token_type_name
                all_token_type_to_output_image_fn_list.setdefault(tmp_token_image_fn_key, []).append(output_image_fn)
                token_type_to_exp_type_to_step_fn_list.setdefault(tmp_token_image_fn_key, {}).setdefault(exp_type, []).append(output_image_fn)

                visualize_token_lens_three_ans_to_layer_result(
                    token_name_to_layer_result,
                    exp_type, tmp_token_type_name,
                    target_answer_idx,
                    input_omit_early_layers,
                    output_image_fn,
                )

            progress.update(1)

        # Merge the images of three answer steps into one (Comment these out if you don't need this)
        if not input_omit_early_layers:
            for token_type, all_output_fn in all_token_type_to_output_image_fn_list.items():
                if exp_type == "token_lens":
                    title = f"Token Lens Logits when Attending to {reformat_var_name_for_visualization(token_type)}: {model_name_for_visualize} on {dataset_name_for_visualize}"
                else:
                    # Knockout
                    title = f"MLP Logit Diff when Knocking Out {reformat_var_name_for_visualization(token_type)}: {model_name_for_visualize} on {dataset_name_for_visualize}"
                merge_figures(
                    all_output_fn, title,
                    f"{args.result_dir}/figures/data_model_specific/full_figures/{input_dataset_name}/{input_model_name}/prompt_template_{input_template_idx}/{token_type}/{exp_type}_logit.png",
                    "token_lens_attn_knockout",
                )

    # Merge all six image (three for token lens and three for knockout) for each specified token (Comment these out if you don't need this)
    if input_omit_early_layers:
        for token_type, result_dict in token_type_to_exp_type_to_step_fn_list.items():
            tmp_fn_list = [fn for fn in result_dict["token_lens"]] + [fn for fn in result_dict["attn_knockout"]]
            token_type_for_visualize = reformat_var_name_for_visualization(token_type)
            merge_dataset_model_specific_token_level_figures(
                tmp_fn_list,
                f"{model_name_for_visualize} on {dataset_name_for_visualize}",
                f"Token Lens Logits when Attending to {token_type_for_visualize}",
                f"MLP Logit Diff: Knockout {token_type_for_visualize}",
                f"{args.result_dir}/figures/data_model_specific/omit_early_layer_figures/{input_dataset_name}/{input_model_name}/prompt_template_{input_template_idx}/{token_type}_logit.png",
            )

def process_critical_tokens_macro_avg_result(input_result_dir, input_omit_early_layers):
    """
    Visualize macro-average results of all models and datasets
    (Lorena: I know this code looks messy. I'll refactor it later when I have time.)
    """
    progress = tqdm(total=len(EXP_TYPE_TO_COMPONENT_NAME.keys()) * 3 * len(ALL_DATASET_NAMES) * len(ALL_MODEL_NAMES) * TEMPLATE_NUM, desc="Processing macro avg results")
    token_type_to_exp_type_to_step_fn_list = {}
    for exp_type in EXP_TYPE_TO_COMPONENT_NAME.keys():
        component_name = EXP_TYPE_TO_COMPONENT_NAME[exp_type]
        for target_answer_idx in [1, 2, 3]:
            # Cache macro avg results (Adjust this according to your needs)
            tmp_all_token_types = STEP_TO_TOKEN_TYPE_NAME[f"{target_answer_idx}"]
            tmp_token_result_fn_list = [f"{exp_type}_{token_type}_results.jsonl" for token_type in tmp_all_token_types]
            dataset_to_exp_type_to_token_layer_result = {tmp_dataset_name: {} for tmp_dataset_name in ALL_DATASET_NAMES}

            for tmp_dataset_name in ALL_DATASET_NAMES:
                for tmp_model_name in ALL_MODEL_NAMES:
                    tmp_model_template_to_token_layer_result = {}
                    # Average within each template
                    for template_idx in [1, 2, 3]:
                        type_to_token_layer_result = {type_name: {token_name: [] for token_name in ALL_TOKENS_TO_BE_VISUALIZED} for type_name in tmp_token_result_fn_list}
                        type_to_df = {tmp_token_result_fn: read_jsonl_to_csv(f"{input_result_dir}/{tmp_dataset_name}/{tmp_model_name}/prompt_template_{template_idx}/{target_answer_idx}/{tmp_token_result_fn}") for tmp_token_result_fn in tmp_token_result_fn_list}
                        for tmp_token_result_fn, tmp_token_type_name in zip(tmp_token_result_fn_list, tmp_all_token_types):
                            for layer_idx in range(LAYER_NUM):
                                if exp_type == "attn_knockout":
                                    # Attention Knockout/MLP
                                    tmp_layer_three_answer_attn_knockout_results = np.array(type_to_df[tmp_token_result_fn][type_to_df[tmp_token_result_fn]["layer_idx"] == layer_idx][f"three_answer_first_token_attn_knockout_{component_name}_logit"].tolist()).transpose()  # shape: (3 answers, num_samples)
                                    tmp_layer_three_answer_full_component_results = np.array(type_to_df[tmp_token_result_fn][type_to_df[tmp_token_result_fn]["layer_idx"] == layer_idx][f"three_answer_first_token_full_{component_name}_logit"].tolist()).transpose()  # shape: (3 answers, num_samples)

                                    tmp_layer_subject_attn_knockout_results = np.array(type_to_df[tmp_token_result_fn][type_to_df[tmp_token_result_fn]["layer_idx"] == layer_idx][f"subject_first_token_attn_knockout_{component_name}_logit"].tolist())  # shape: (num_samples,)
                                    tmp_layer_subject_full_component_results = np.array(type_to_df[tmp_token_result_fn][type_to_df[tmp_token_result_fn]["layer_idx"] == layer_idx][f"subject_first_token_full_{component_name}_logit"].tolist())  # shape: (num_samples,)

                                    for tmp_ans_idx in [1, 2, 3]:
                                        type_to_token_layer_result[tmp_token_result_fn][f'answer_{tmp_ans_idx}'].append(np.average(tmp_layer_three_answer_full_component_results[tmp_ans_idx - 1] - tmp_layer_three_answer_attn_knockout_results[tmp_ans_idx - 1], axis=0))
                                    type_to_token_layer_result[tmp_token_result_fn]['subject'].append(np.average(tmp_layer_subject_full_component_results - tmp_layer_subject_attn_knockout_results, axis=0))
                                else:
                                    # Token lens/Attention
                                    tmp_layer_three_answer_results = np.array(type_to_df[tmp_token_result_fn][type_to_df[tmp_token_result_fn]["layer_idx"] == layer_idx][f"three_answer_first_token_logit"].tolist()).transpose()  # shape: (3 answers, num_samples)
                                    tmp_layer_subject_results = np.array(type_to_df[tmp_token_result_fn][type_to_df[tmp_token_result_fn]["layer_idx"] == layer_idx][f"subject_first_token_logit"].tolist())  # shape: (num_samples,)

                                    for tmp_ans_idx in [1, 2, 3]:
                                        type_to_token_layer_result[tmp_token_result_fn][f'answer_{tmp_ans_idx}'].append(np.average(tmp_layer_three_answer_results[tmp_ans_idx - 1], axis=0))
                                    type_to_token_layer_result[tmp_token_result_fn]['subject'].append(np.average(tmp_layer_subject_results, axis=0))

                        progress.update(1)

                        if tmp_model_template_to_token_layer_result == {}:  # Initialize
                            tmp_model_template_to_token_layer_result = type_to_token_layer_result

                        # Add up avg from each template
                        for type_name, result in type_to_token_layer_result.items():
                            for token_name, result_list in result.items():
                                tmp_model_template_to_token_layer_result[type_name][token_name] = (np.array(tmp_model_template_to_token_layer_result[type_name][token_name]) + np.array(result_list)).tolist()

                    # Macro-average across templates
                    for type_name, result in tmp_model_template_to_token_layer_result.items():
                        for token_name, result_list in result.items():
                            if dataset_to_exp_type_to_token_layer_result[tmp_dataset_name] == {}:
                                dataset_to_exp_type_to_token_layer_result[tmp_dataset_name] = {type_name: {token_name: None for token_name in ALL_TOKENS_TO_BE_VISUALIZED} for type_name in tmp_token_result_fn_list}
                            if dataset_to_exp_type_to_token_layer_result[tmp_dataset_name][type_name][token_name] is None:
                                dataset_to_exp_type_to_token_layer_result[tmp_dataset_name][type_name][token_name] = (np.array(result_list) / TEMPLATE_NUM).tolist()
                            else:
                                dataset_to_exp_type_to_token_layer_result[tmp_dataset_name][type_name][token_name] = np.sum([dataset_to_exp_type_to_token_layer_result[tmp_dataset_name][type_name][token_name], np.array(result_list) / TEMPLATE_NUM], axis=0).tolist()

                # Macro-average across models
                for type_name, result in dataset_to_exp_type_to_token_layer_result[tmp_dataset_name].items():
                    for token_name, result_list in result.items():
                        dataset_to_exp_type_to_token_layer_result[tmp_dataset_name][type_name][token_name] = (np.array(result_list) / len(ALL_MODEL_NAMES)).tolist()

            # Macro-average across datasets
            macro_avg_result = {type_name: {token_to_be_visualized: None for token_to_be_visualized in ALL_TOKENS_TO_BE_VISUALIZED} for type_name in tmp_token_result_fn_list}
            for _, result_dict in dataset_to_exp_type_to_token_layer_result.items():
                for type_name, result in result_dict.items():
                    for token_name, token_vals in result.items():
                        if macro_avg_result[type_name][token_name] is None:
                            macro_avg_result[type_name][token_name] = np.array(token_vals)
                        else:
                            macro_avg_result[type_name][token_name] += np.array(token_vals)
            for type_name, result_dict in macro_avg_result.items():
                for token_name, token_vals in result_dict.items():
                    macro_avg_result[type_name][token_name] = (token_vals / len(ALL_DATASET_NAMES)).tolist()

            for tmp_token_result_fn, tmp_token_type_name in zip(tmp_token_result_fn_list, tmp_all_token_types):
                tmp_output_dir = os.path.join(f"{input_result_dir}/figures/avg_figures", f"{'full_figures' if not input_omit_early_layers else 'omit_early_layer_figures'}")
                os.makedirs(tmp_output_dir, exist_ok=True)
                if tmp_token_type_name == "answer_1" or tmp_token_type_name == "answer_2":
                    token_name = "prev_ans"
                    output_fn = f"{tmp_output_dir}/{token_name}/{exp_type}_{tmp_token_type_name}_at_step_#{target_answer_idx}.png"
                else:
                    token_name = tmp_token_type_name
                    output_fn = f"{tmp_output_dir}/{token_name}/{exp_type}_at_step_#{target_answer_idx}.png"
                visualize_token_lens_three_ans_to_layer_result(
                    macro_avg_result[tmp_token_result_fn],
                    exp_type, tmp_token_type_name,
                    target_answer_idx,
                    input_omit_early_layers,
                    output_fn,
                )
                token_type_to_exp_type_to_step_fn_list.setdefault(token_name, {}).setdefault(exp_type, []).append(output_fn)

        # Merge the images of three answer steps into one (Comment these out if you don't need this)
        if not input_omit_early_layers:
            for token_type, all_output_fn in token_type_to_exp_type_to_step_fn_list.items():
                if exp_type == "token_lens":
                    title = f"Token Lens Logits when Attending to {reformat_var_name_for_visualization(token_type)}"
                else:
                    # Knockout
                    title = f"MLP Logit Diff when Knocking Out {reformat_var_name_for_visualization(token_type)}"
                merge_figures(
                    all_output_fn[exp_type], title,
                    f"{input_result_dir}/figures/avg_figures/full_figures/{token_type}/{exp_type}_logit.png",
                    "token_lens_attn_knockout",
                )

    if input_omit_early_layers:
        for token_type, result_dict in token_type_to_exp_type_to_step_fn_list.items():
            tmp_fn_list = [fn for fn in result_dict["token_lens"]] + [fn for fn in result_dict["attn_knockout"]]
            token_type_for_visualize = reformat_var_name_for_visualization(token_type)
            merge_dataset_model_specific_token_level_figures(
                tmp_fn_list,
                f"",
                f"Token Lens Logits when Attending to {token_type_for_visualize}",
                f"MLP Logit Diff: Knockout {token_type_for_visualize}",
                f"{input_result_dir}/figures/avg_figures/omit_early_layer_figures/{token_type}_logit.png",
            )

if __name__ == "__main__":
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument("--result_dir", type=str, required=True)
    args_parser.add_argument("--dataset_name", type=str)
    args_parser.add_argument("--model_name", type=str)
    args_parser.add_argument("--template_idx", type=str, required=True)  # ["1", "2", "3", "macro_avg"]
    args_parser.add_argument("--omit_early_layers", type=lambda x:x.lower()=="true", required=True)  # True for reproducing figure with 6 subfigures in a row in the paper. False for full figures in the appendix
    args = args_parser.parse_args()

    LAYER_NUM = 32
    OMIT_LAYER_START_IDX = 15

    # When visualizing macro average results across all models, datasets, and prompt templates, all or none of dataset_name, model_name, template_idx should be "macro_avg"
    # Otherwise, choose specific dataset_name, model_name, and template_idx
    assert (args.dataset_name == "macro_avg") == (args.model_name == "macro_avg") == (args.template_idx == "macro_avg")
    if args.dataset_name == "macro_avg":
        print(f"Visualize results in {args.result_dir}")
        process_critical_tokens_macro_avg_result(args.result_dir, input_omit_early_layers=args.omit_early_layers)
    else:
        print(f"Visualize results in {args.result_dir}/{args.dataset_name}/{args.model_name}")
        process_dataset_model_specific_critical_tokens_result(
            input_result_dir=args.result_dir,
            input_dataset_name=args.dataset_name,
            input_model_name=args.model_name,
            input_template_idx=args.template_idx,
            input_omit_early_layers=args.omit_early_layers
        )

    #bash scripts/visualize_critical_token_outputs.sh
