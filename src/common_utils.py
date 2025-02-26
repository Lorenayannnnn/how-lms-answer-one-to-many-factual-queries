
import os

from omegaconf import OmegaConf

STEP_TO_TOKEN_TYPE_NAME = {
    "1": ["subject", "last_token"],
    "2": ["subject", "answer_1", "last_token"],
    "3": ["subject", "answer_1", "answer_2", "last_token"],
}

ALL_DATASET_NAMES = ["country_cities", "artist_songs", "actor_movies"]

ALL_MODEL_NAMES = ["meta-llama/Meta-Llama-3-8B-Instruct", "mistralai/Mistral-7B-Instruct-v0.2"]

ANS_TO_LATEX = {"Answer 1": r"$o^{(1)}$", "Answer 2": r"$o^{(2)}$"}

def prepare_folder(file_path):
    """Prepare a folder for a file"""
    folder = os.path.dirname(file_path)
    if not os.path.exists(folder):
        os.makedirs(folder)


def load_config_and_setup_output_dir(args):
    base_configs = args.base_configs
    if not os.path.exists(args.base_configs):
        raise FileNotFoundError(f"Config file {args.base_configs} does not exist")
    configs = OmegaConf.load(base_configs)
    target_answer_idx = configs.running_args.target_answer_idx

    short_dataset_name = configs.data_args.dataset_name
    configs.data_args.dataset_name = f"datasets/{short_dataset_name}/{configs.model_args.model_name_or_path}/{short_dataset_name}_{target_answer_idx}.jsonl"

    configs.running_args.output_dir = f"outputs/{configs.running_args.exp_type}/{short_dataset_name}/{configs.model_args.model_name_or_path}/{target_answer_idx}"

    output_dir = configs.running_args.output_dir

    output_config_fn = os.path.join(output_dir, "configs.yaml")
    prepare_folder(output_config_fn)
    OmegaConf.save(configs, output_config_fn)

    return configs
