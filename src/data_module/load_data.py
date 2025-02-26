
from torch.utils.data import DataLoader

from datasets import load_dataset, Dataset

from src.data_module.OneToManyDataCollator import OneToManyDataCollator


def load_data_from_hf(file_or_dataset_name, cache_dir=None):
    raw_datasets = load_dataset(
        "json",
        data_files=file_or_dataset_name,
        cache_dir=cache_dir,
    )
    return raw_datasets["train"]


def setup_dataloader(tokenized_dataset, batch_size, tokenizer):
    return DataLoader(tokenized_dataset, batch_size=batch_size, collate_fn=OneToManyDataCollator(tokenizer, return_tensors="pt"))
