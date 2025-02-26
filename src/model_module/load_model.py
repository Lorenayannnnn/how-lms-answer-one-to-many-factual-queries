
from src.model_module.my_modeling_llama import HookedLlamaForCausalLM
from src.model_module.my_modeling_mistral import HookedMistralForCausalLM


def load_model(configs):
    """main function for loading the model_module"""
    if "Llama" in configs.model_args.model_name_or_path:
        model = HookedLlamaForCausalLM.from_pretrained(
            configs.model_args.model_name_or_path,
            cache_dir=configs.data_args.cache_dir,
            device_map="auto",
            torch_dtype="auto",
        )
        print("Loaded HookedLlamaForCausalLM | ", 'dtype:', next(model.parameters()).dtype)
    elif "Mistral" in configs.model_args.model_name_or_path:
        model = HookedMistralForCausalLM.from_pretrained(
            configs.model_args.model_name_or_path,
            cache_dir=configs.data_args.cache_dir,
            device_map="auto",
            torch_dtype="auto"
        )
        print("Loaded HookedMistralForCausalLM | ", 'dtype:', next(model.parameters()).dtype)
    else:
        raise ValueError("Model not supported")
    return model
