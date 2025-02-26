import argparse
import json
import os

import torch
from tqdm import tqdm

from src.causal_tracing.utils import ModelAndTokenizer, get_next_token_and_prob, \
    make_inputs, trace_with_patch, decode_tokens, layername, plot_trace_heatmap, collect_embedding_std


def calculate_hidden_flow(
    mt, prompt, tokens_to_be_noised_start_end_idx, samples=10, noise=0.1, window=10, kind=None, target_answer_tokens: list=None
):
    """
    Runs causal tracing over every token/layer combination in the network
    and returns a dictionary numerically summarizing the results.
    Currently only support greedy decoding
    """
    inp = make_inputs(mt.tokenizer, [prompt] * (samples + 1))
    with torch.no_grad():
        answer_t, base_score = get_next_token_and_prob(mt, inp, target_answer_tokens)

    e_range = tokens_to_be_noised_start_end_idx
    low_score = trace_with_patch(
        mt, inp, [], answer_t, e_range, noise=noise,
    ).item()
    if not kind:
        differences = trace_important_states(
            mt, mt.num_layers, inp, e_range, answer_t, noise=noise
        )
    else:
        differences = trace_important_window(
            mt,
            mt.num_layers,
            inp,
            e_range,
            answer_t,
            noise=noise,
            window=window,
            kind=kind,
        )
    differences = differences.detach().cpu()
    answer = mt.tokenizer.decode([answer_t])
    return dict(
        scores=differences,
        low_score=low_score,
        high_score=base_score,
        input_ids=inp["input_ids"][0],
        input_tokens=decode_tokens(mt.tokenizer, inp["input_ids"][0]),
        subject_range=e_range,
        answer=answer,
        window=window,
        kind=kind or "",
    )


def trace_important_states(mt, num_layers, inp, e_range, answer_t, noise=0.1):
    # Do restorations all the way to the position before the first answer token
    ntoks = inp["input_ids"].shape[1]

    table = []
    for tnum in range(ntoks):
        row = []
        # Use the token that was predicted as the target token
        for layer in range(0, num_layers):
            r = trace_with_patch(
                mt,
                inp,
                [(tnum, layername(mt.model, layer))],
                answer_t,
                tokens_to_mix=e_range,
                noise=noise,
            )
            row.append(r)
        table.append(torch.stack(row))
    return torch.stack(table)


def trace_important_window(
    mt, num_layers, inp, e_range, answer_t, kind, window=10, noise=0.1
):
    ntoks = inp["input_ids"].shape[1]
    table = []
    for tnum in range(ntoks):
        row = []
        for layer in range(0, num_layers):
            layerlist = [
                (tnum, layername(mt.model, L, kind))
                for L in range(
                    max(0, layer - window // 2), min(num_layers, layer - (-window // 2))
                )
            ]
            r = trace_with_patch(
                mt, inp, layerlist, answer_t, tokens_to_mix=e_range, noise=noise
            )
            row.append(r)
        table.append(torch.stack(row))
    return torch.stack(table)


def has_processed(output_tracking_fn, name_to_check):
    if not os.path.exists(output_tracking_fn): return False
    with open(output_tracking_fn, "r") as f_tracking:
        result_to_save = json.load(f_tracking)
        return name_to_check in result_to_save

def plot_all_flow(mt, prompt, tokens_to_be_noised_start_end_idx, noise, output_dir=None,
                  instance_name=None, target_answer_tokens: list = None, other_info=None):
    """
    :param mt: model tokenizer object
    :param prompt: input query
    :param tokens_to_be_noised_start_end_idx: start and end index of the tokens to be noised (subject or a previous answer)
    :param noise: noise level
    :param output_dir: {pred_result_dir}/causal_tracing_{examine_type}/noise_subject_step_{pred_step}
    :param instance_name: {output_dir}/{subject}_pred_{target_pred}
    :param target_answer_tokens
    :param other_info: original pred line
    """
    for kind in ["mlp", "attn"]:
        instance_kind_name = f"{instance_name}_{kind}"
        # Read tracking json and see if the intervention has been done (Track progress for easier restart)
        examine_type_step_kind_output_path = os.path.join(output_dir, f"{kind}_result")
        output_tracking_fn = f"{examine_type_step_kind_output_path}_tracking.jsonl"
        if not has_processed(output_tracking_fn, instance_kind_name):
            result = calculate_hidden_flow(mt, prompt, tokens_to_be_noised_start_end_idx, noise=noise, kind=kind,
                                           target_answer_tokens=target_answer_tokens)
            # We are only saving the result here and not the plot for individual instance (will calculate avg later)
            plot_trace_heatmap(result, instance_name, save_result=True,
                               examine_type_step_kind_output_path=examine_type_step_kind_output_path, other_info=other_info)
        print("Finished ", instance_kind_name)


def get_model_task_noise(project_root_dir, model_name, dataset_name, examine_type):
    # examine_type: noise_subject, noise_prev_ans
    noise_dict_json_fn = f"{project_root_dir}/datasets/dataset_name_to_model_name_to_subject_answers_noise_dict.json"
    with open(noise_dict_json_fn) as f:
        noise_dict = json.load(f)
    return noise_dict[dataset_name][model_name]["subject" if examine_type == "noise_subject" else "answers"]


def calculate_noise_level(all_pred_lines, examine_type, mt):
    # examine_type: noise_subject, noise_prev_ans
    all_tokens_for_calculating_noise = []
    for c in all_pred_lines:
        line = json.loads(c)
        if examine_type == "noise_subject":
            all_tokens_for_calculating_noise.append(line["subject"])
        else:
            # noise prev ans
            all_tokens_for_calculating_noise.extend(line["three_answers_label_list"])
    return 3 * collect_embedding_std(mt, all_tokens_for_calculating_noise)


def causal_analysis_on_one_to_many_pred_results(input_args):
    project_root_dir = input_args.project_root_dir
    model_name = input_args.model_name
    dataset_name = input_args.dataset_name
    max_causal_trace_cnt = 100      # Causal tracing using 100 entries

    mt = ModelAndTokenizer(
        model_name,
        cache_dir=input_args.cache_dir,
        torch_dtype="auto"
    )

    examine_type = input_args.examine_type
    pred_result_dir = f"{project_root_dir}/datasets/{dataset_name}/{model_name}"
    output_dir = pred_result_dir + f"/causal_tracing_{examine_type}"
    os.makedirs(output_dir, exist_ok=True)

    for pred_step in [1, 2, 3]:
        with open(os.path.join(pred_result_dir, f"{dataset_name}_{pred_step}.jsonl")) as f:
            all_pred_results = f.readlines()[:max_causal_trace_cnt]
            pred_result_lines = [json.loads(c) for c in all_pred_results]
        if examine_type == "noise_subject":
            tmp_output_dir = os.path.join(output_dir, f"noise_subject_step_{pred_step}")
            os.makedirs(tmp_output_dir, exist_ok=True)
            # Noise for each model and dataset has already been provided in the datasets dir. You can recalculate if you want
            # noise_level = calculate_noise_level(all_pred_results, examine_type, mt)
            noise_level = get_model_task_noise(project_root_dir, model_name, dataset_name, examine_type)
            for idx, pred_line in tqdm(enumerate(pred_result_lines)):
                subject = pred_line['subject']
                # Answers
                three_answers_label_list = pred_line['three_answers_label_list']
                target_pred = three_answers_label_list[pred_step - 1]
                target_answer_tokens = pred_line["target_answer_tokens"]
                plot_all_flow(mt=mt, prompt=pred_line["query"], tokens_to_be_noised_start_end_idx=pred_line["subject_start_end_idx"],
                              noise=noise_level, output_dir=tmp_output_dir,
                              instance_name=f"{tmp_output_dir}/{subject}_pred_{target_pred}",
                              target_answer_tokens=target_answer_tokens, other_info=pred_line)
        elif examine_type == "noise_prev_ans":
            if pred_step == 1:
                print("No prev ans for pred_step 1")
                continue
            # Noise for each model and dataset has already been provided in the datasets dir. You can recalculate if you want
            # noise_level = calculate_noise_level(all_pred_results, examine_type, mt)
            noise_level = get_model_task_noise(project_root_dir, model_name, dataset_name, examine_type)
            print(f"Using noise level {noise_level}")
            for idx, pred_line in tqdm(enumerate(pred_result_lines)):
                subject = pred_line["subject"]
                three_answers_label_list = pred_line['three_answers_label_list']
                target_answer_tokens = pred_line["target_answer_tokens"]
                curr_ans_index = pred_step - 1
                target_pred = three_answers_label_list[curr_ans_index]
                if pred_step >= 2:
                    # Noise o1 at step 2, or o2 at step 3
                    tmp_output_dir = os.path.join(output_dir, f"pred_{pred_step}_noise_{pred_step-1}")
                    os.makedirs(tmp_output_dir, exist_ok=True)
                    answer_tokens_to_be_noised_start_end = pred_line["three_answers_start_end_idx"][curr_ans_index - 1]
                    plot_all_flow(mt=mt, prompt=pred_line["query"], tokens_to_be_noised_start_end_idx=answer_tokens_to_be_noised_start_end,
                                  noise=noise_level, output_dir=tmp_output_dir,
                                  instance_name=f"{tmp_output_dir}/{subject}_pred_{target_pred}",
                                  target_answer_tokens=target_answer_tokens, other_info=pred_line)
                if pred_step == 3:
                    # Noise o1 at step 3
                    tmp_output_dir = os.path.join(output_dir, f"pred_{pred_step}_noise_{pred_step-2}")
                    os.makedirs(tmp_output_dir, exist_ok=True)
                    answer_tokens_to_be_noised_start_end = pred_line["three_answers_start_end_idx"][curr_ans_index-2]
                    plot_all_flow(mt=mt, prompt=pred_line["query"], tokens_to_be_noised_start_end_idx=answer_tokens_to_be_noised_start_end,
                                  noise=noise_level, output_dir=tmp_output_dir,
                                  instance_name=f"{tmp_output_dir}/{subject}_pred_{target_pred}",
                                  target_answer_tokens=target_answer_tokens, other_info=pred_line)
        else:
            raise ValueError(f"Unknown examine type {examine_type}")


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--project_root_dir", type=str, required=True)
    arg_parser.add_argument("--examine_type", type=str, required=True)
    arg_parser.add_argument("--model_name", type=str, required=True)
    arg_parser.add_argument("--dataset_name", type=str, required=True)
    arg_parser.add_argument("--cache_dir", type=lambda x: None if x == "" else x, required=False, default=None)
    args = arg_parser.parse_args()

    causal_analysis_on_one_to_many_pred_results(args)

    # bash scripts/run_causal_tracing.sh