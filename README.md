# how-lms-answer-one-to-many-factual-queries
This repository is the official implementation of our EMNLP 2025 paper [Promote, Suppress, Iterate: How Language Models Answer One-to-Many Factual Queries](https://www.arxiv.org/abs/2502.20475).

## Setup
### Dependencies
Experiments are run in:

|    Package     | Version |
|:--------------:|:-------:|
|     conda      | 23.1.0  |
|     Python     |   3.8   |
|      CUDA      |  11.7   |

### Install
```bash
conda create -n one_to_many python=3.8
conda activate one_to_many
pip install -r requirements.txt
```

## Datasets
The [datasets](datasets) directory contains all the data used in the experiments with the following structure:
```
└── datasets
    └── actor_movies
        └──meta-llama/Meta-Llama-3-8B-Instruct
            └──prompt_template_1
               ├── actor_movies_1.jsonl
               ├── actor_movies_2.jsonl
               └── actor_movies_3.jsonl
            └──prompt_template_2
            └──prompt_template_3
        └──mistralai/Mistral-7B-Instruct-v0.2
            └── ...
    └── artist_songs
    └── country_cities
```
Each pair of dataset and model has three prompt templates. $i$ in ```{dataset_name}_{i}.jsonl``` is the answer step. The datasets are also available on Hugging Face at [LorenaYannnnn/how_lms_answer_one_to_many_factual_queries](https://huggingface.co/datasets/LorenaYannnnn/how_lms_answer_one_to_many_factual_queries).

Refer to [README.md](datasets/README.md) for file content details and section 3.2 of the paper for dataset curation.

## Experiments
### Decoding the Overall Mechanism: Decode Attention and MLP Outputs
1. Set model, dataset, prompt template, and answer step in [run_main_configs.yaml](configs/run_main_configs.yaml). Set ```exp_type``` to be ```decode_attn_mlp_outputs```.
2. Run:
    ```
    bash scripts/run_main.sh
    ```
3. Visualize results by setting model/dataset in [visualize_decode_attention_mlp_output.sh](scripts/visualize_decode_attention_mlp_output.sh) and run:
    ```
    bash scripts/visualize_decode_attention_mlp_output.sh
    ```
   Adjust visualization parameters in ```visualize_token_lens_three_ans_to_layer_result()``` in [visualize_decode_attention_mlp_outputs.py](src/analysis_module/visualize_decode_attention_mlp_outputs.py).

### Which Tokens Matter: Causal Tracing
We follow [ROME](https://github.com/kmeng01/rome) to run causal tracing experiments. All codes are in [causal_tracing](src/causal_tracing) directory.
1. Set model, dataset, prompt template, and token you want to noise (noise_subject/noise_prev_ans) in [run_causal_tracing.sh](scripts/run_causal_tracing.sh).
2. Run:
    ```
    bash scripts/run_causal_tracing.sh
    ```
3. Visualize results by setting model/dataset in [visualize_causal_tracing.sh](scripts/visualize_causal_tracing.sh) and run:
    ```
    bash scripts/visualize_causal_tracing.sh
    ```

### Critical Token Analysis: Token Lens & Attention Knockout
1.  Set model, dataset, and answer step in [run_main_configs.yaml](configs/run_main_configs.yaml). Set ```exp_type``` to be ```analyze_critical_tokens```.
2. Run:
    ```
    bash scripts/run_main.sh
    ```
3. Visualize results by setting model/dataset in [visualize_critical_token_outputs.sh](scripts/visualize_critical_token_outputs.sh) and run:
    ```
    bash scripts/visualize_critical_token_outputs.sh
    ```

### Function of Attention Heads (Appendix F)
1. Set ```exp_type``` to be ```examine_finegrained_attention_head``` in [run_main_configs.yaml](configs/run_main_configs.yaml)
2. Run:
      ```
      bash scripts/run_main.sh
      ```
3. Visualize results by setting model/dataset in [visualize_finegrained_attention_head_output.sh](scripts/visualize_finegrained_attention_head_output.sh) and run:
      ```
      bash scripts/visualize_finegrained_attention_head_output.sh
      ```

## Citation
```
@inproceedings{yan2025promote,
  title     = {Promote, Suppress, Iterate: How Language Models Answer One-to-Many Factual Queries},
  author    = {Yan, Tianyi Lorena and Jia, Robin},
  booktitle = {Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing},
  year      = {2025}
}
```