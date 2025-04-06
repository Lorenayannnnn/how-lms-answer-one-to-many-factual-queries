
import torch
from transformers import AutoTokenizer


def tokenize(entry, target_answer_idx):
    # Change this according to your needs
    unused_columns = ["target_answer_idx", "relation_start_end_idx"]
    for col in unused_columns:
        if col in entry.keys():
            entry.pop(col)

    entry["input_ids"] = entry.pop("query_input_ids")
    entry["attention_mask"] = torch.ones(len(entry["input_ids"]))
    entry["subject_first_token"] = entry.pop("subject_token_list")[0]
    three_answers_token_list = entry.pop("three_answers_token_list")
    entry["three_answers_first_token_list"] = [three_answers_token_list[i][0] for i in range(target_answer_idx)]
    three_answers_start_end_idx = entry.pop("three_answers_start_end_idx")

    # Calculate position from right in case of padding from left
    entry["subject_last_token_position_from_right"] = -(len(entry["input_ids"]) - entry["subject_start_end_idx"][1]) - 1

    for tmp_idx in range(1, target_answer_idx):
        entry[f"right_to_answer_{tmp_idx}_last_token"] = -(len(entry["input_ids"]) - three_answers_start_end_idx[tmp_idx - 1][1]) - 1
        entry[f"prev_answer_{tmp_idx}_start_end_idx"] = three_answers_start_end_idx[tmp_idx - 1]

    # if "prev_answer_1_start_end_idx" in entry.keys():
    #     entry["right_to_answer_1_last_token"] = -(len(entry["input_ids"]) - entry["prev_answer_1_start_end_idx"][1]) - 1
    # if "prev_answer_2_start_end_idx" in entry.keys():
    #     entry["right_to_answer_2_last_token"] = -(len(entry["input_ids"]) - entry["prev_answer_2_start_end_idx"][1]) - 1

    entry["last_token_start_end_idx"] = [len(entry["input_ids"]) - 1, len(entry["input_ids"])]
    entry["last_token_position_from_right"] = -1

    return entry


def preprocess(configs, predict_dataset):
    """takes in the raw dataset and preprocesses it into the format we want"""

    tokenizer = create_tokenizer(configs)
    tokenized_dataset = predict_dataset.map(tokenize)

    return tokenizer, tokenized_dataset

def create_tokenizer(configs):
    """creates the tokenizer"""
    tokenizer = AutoTokenizer.from_pretrained(
        configs.model_args.tokenizer_name if configs.model_args.tokenizer_name else configs.model_args.model_name_or_path,
        padding_side="left",
        add_bos_token=configs.data_args.add_bos_token,
        add_eos_token=configs.data_args.add_eos_token,
        cache_dir=configs.data_args.cache_dir,
    )
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer

