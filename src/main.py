import argparse

from src.common_utils import load_config_and_setup_output_dir
from src.data_module.load_data import load_data_from_hf, setup_dataloader
from src.data_module.preprocessing import preprocess
from src.model_module.load_model import load_model
from src.train_module.OneToManyExpRunner import OneToManyExpRunner


def main(args):
    print("Start running...")
    configs = load_config_and_setup_output_dir(args)

    """Load the data"""
    raw_datasets = load_data_from_hf(configs.data_args.dataset_name, cache_dir=configs.data_args.cache_dir)

    """Preprocess data"""
    tokenizer, tokenized_dataset = preprocess(configs, raw_datasets)
    data_loader = setup_dataloader(tokenized_dataset=tokenized_dataset, batch_size=configs.running_args.batch_size, tokenizer=tokenizer)

    """Load model"""
    model = load_model(configs)

    """Set up trainer"""
    exp_runner = OneToManyExpRunner(model=model, tokenizer=tokenizer, data_loader=data_loader, args=configs.running_args)

    print(f"Running {configs.running_args.exp_type} | target_answer_idx: {configs.running_args.target_answer_idx} | model_name: {configs.model_args.model_name_or_path} | dataset_name: {configs.data_args.dataset_name}")
    if configs.running_args.exp_type == "decode_attn_mlp_outputs":
        exp_runner.do_decode_attention_mlp()
    elif configs.running_args.exp_type == "analyze_critical_tokens":
        exp_runner.do_token_lens_and_attention_knockout()
    elif configs.running_args.exp_type == "examine_finegrained_attention_head":
        exp_runner.examine_finegrained_attention_head()
    else:
        raise ValueError("Invalid experiment type. Choose from 'decode_attn_mlp_outputs', 'analyze_critical_tokens', 'examine_finegrained_attention_head'")
    print("Finished")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base_configs",
        type=str,
        required=True,
        help="Base configuration file"
    )
    args = parser.parse_args()
    main(args)
