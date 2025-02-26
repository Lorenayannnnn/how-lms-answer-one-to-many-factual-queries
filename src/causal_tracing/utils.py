import json
import os
import re
from collections import defaultdict

import numpy
import torch
from matplotlib import pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image

from src.causal_tracing import nethook
from src.model_module.my_modeling_llama import HookedLlamaForCausalLM
from src.model_module.my_modeling_mistral import HookedMistralForCausalLM


class ModelAndTokenizer:
    """
    An object to hold on to (or automatically download and hold)
    a GPT-style language model and tokenizer.  Counts the number
    of layers.
    """

    def __init__(
        self,
        model_name=None,
        model=None,
        tokenizer=None,
        low_cpu_mem_usage=False,
        torch_dtype=None,
        cache_dir=None,
    ):
        if tokenizer is None:
            assert model_name is not None
            tokenizer = AutoTokenizer.from_pretrained(model_name,padding_side="left",add_bos_token=False,add_eos_token=False,cache_dir=cache_dir)
        if model is None:
            assert model_name is not None
            if model_name == "meta-llama/Meta-Llama-3-8B-Instruct":
                model = HookedLlamaForCausalLM.from_pretrained(
                    model_name,
                    cache_dir=cache_dir,
                    device_map="auto",
                    torch_dtype=torch_dtype,
                )
            elif model_name == "mistralai/Mistral-7B-Instruct-v0.2":
                print("Initialize HookedMistralForCausalLM")
                model = HookedMistralForCausalLM.from_pretrained(
                    model_name,
                    cache_dir=cache_dir,
                    device_map="auto",
                    torch_dtype=torch_dtype,
                )
            else:
                model = AutoModelForCausalLM.from_pretrained(
                    model_name, low_cpu_mem_usage=low_cpu_mem_usage, torch_dtype=torch_dtype, cache_dir=cache_dir
                )
            nethook.set_requires_grad(False, model)
            model.eval().cuda()
            # print dtype
            print("model type", type(model), 'model dtype:', next(model.parameters()).dtype)

        self.tokenizer = tokenizer
        self.model = model
        self.layer_names = [n for n, m in model.named_modules() if (re.match(r"^(model)\.(h|layers)\.\d+$", n))]
        self.num_layers = len(self.layer_names)

    def __repr__(self):
        return (
            f"ModelAndTokenizer(model: {type(self.model).__name__} "
            f"[{self.num_layers} layers], "
            f"tokenizer: {type(self.tokenizer).__name__})"
        )


def get_next_token_and_prob(mt, input_dict, target_answer_tokens: list=None):
    clean_inp = {'input_ids': input_dict['input_ids'][0].unsqueeze(0), 'attention_mask': input_dict['attention_mask'][0].unsqueeze(0)}
    outputs = mt.model.run_with_cache(**clean_inp)
    clean_logits = outputs[0].logits
    clean_gen_token = clean_logits[:, -1, :].argmax(-1).tolist()[0]
    clean_prob = torch.softmax(clean_logits[:, -1, :], dim=-1)[0, clean_gen_token].item()
    assert clean_gen_token == target_answer_tokens[0], f"Next token does not match"

    return clean_gen_token, clean_prob


def make_inputs(tokenizer, prompts, device="cuda", add_special_tokens=False):
    token_lists = [tokenizer.encode(p, add_special_tokens=add_special_tokens) for p in prompts]
    maxlen = max(len(t) for t in token_lists)
    if "[PAD]" in tokenizer.all_special_tokens:
        pad_id = tokenizer.all_special_ids[tokenizer.all_special_tokens.index("[PAD]")]
    else:
        pad_id = 0
    input_ids = [[pad_id] * (maxlen - len(t)) + t for t in token_lists]
    attention_mask = [[0] * (maxlen - len(t)) + [1] * len(t) for t in token_lists]
    return dict(
        input_ids=torch.tensor(input_ids).to(device),
        attention_mask=torch.tensor(attention_mask).to(device),
    )


def layername(model, num, kind=None):
    if type(model) == HookedLlamaForCausalLM or type(model) == HookedMistralForCausalLM:
        if kind == "embed":
            return "model.embed_tokens"
        if kind == "attn":
            kind = "self_attn"
        return f'model.layers.{num}{"" if kind is None else "." + kind}'
    assert False, "unknown transformer structure"


def trace_with_patch(
    mt,  # The model + tokenizer
    inp,  # A set of inputs
    states_to_patch,  # A list of (token index, layername) triples to restore
    answers_t,  # Answer probabilities to collect
    tokens_to_mix,  # Range of tokens to corrupt (begin, end)
    noise=0.1,  # Level of noise to add
    uniform_noise=False,
    replace=False,  # True to replace with instead of add noise
    trace_layers=None,  # List of traced outputs to return
):
    """
    Runs a single causal trace.  Given a model and a batch input where
    the batch size is at least two, runs the batch in inference, corrupting
    a the set of runs [1...n] while also restoring a set of hidden states to
    the values from an uncorrupted run [0] in the batch.

    The convention used by this function is that the zeroth element of the
    batch is the uncorrupted run, and the subsequent elements of the batch
    are the corrupted runs.  The argument tokens_to_mix specifies an
    be corrupted by adding Gaussian noise to the embedding for the batch
    inputs other than the first element in the batch.  Alternately,
    subsequent runs could be corrupted by simply providing different
    input tokens via the passed input batch.

    Then when running, a specified set of hidden states will be uncorrupted
    by restoring their values to the same vector that they had in the
    zeroth uncorrupted run.  This set of hidden states is listed in
    states_to_patch, by listing [(token_index, layername), ...] pairs.
    To trace the effect of just a single state, this can be just a single
    token/layer pair.  To trace the effect of restoring a set of states,
    any number of token indices and layers can be listed.
    """
    model = mt.model
    rs = numpy.random.RandomState(1)  # For reproducibility, use pseudorandom noise
    if uniform_noise:
        prng = lambda *shape: rs.uniform(-1, 1, shape)
    else:
        prng = lambda *shape: rs.randn(*shape)

    patch_spec = defaultdict(list)
    for t, l in states_to_patch:
        patch_spec[l].append(t)

    embed_layername = layername(model, 0, "embed")

    def untuple(x):
        return x[0] if isinstance(x, tuple) else x

    # Define the model-patching rule.
    if isinstance(noise, float):
        noise_fn = lambda x: noise * x
    else:
        noise_fn = noise

    def patch_rep(x, layer):
        if layer == embed_layername:
            # If requested, we corrupt a range of token embeddings on batch items x[1:]
            if tokens_to_mix is not None:
                b, e = tokens_to_mix
                noise_data = noise_fn(
                    torch.from_numpy(prng(x.shape[0] - 1, e - b, x.shape[2]))
                ).to(x.device)
                if replace:
                    x[1:, b:e] = noise_data
                else:
                    x[1:, b:e] += noise_data
            return x
        if layer not in patch_spec:
            return x
        # If this layer is in the patch_spec, restore the uncorrupted hidden state
        # for selected tokens.
        h = untuple(x)
        for t in patch_spec[layer]:
            # h shape: (batch_size, seq_len, hidden_size)
            # batch = 0 is the uncorrupted run
            # Reset hidden state for the token at position t for the current specified layer (corresponds to resetting last couple of layers of each row in the paper's figure 1)
            h[1:, t] = h[0, t]
        return x

    # With the patching rules defined, run the patched model in inference.
    additional_layers = [] if trace_layers is None else trace_layers
    with torch.no_grad(), nethook.TraceDict(
        model,
        [embed_layername] + list(patch_spec.keys()) + additional_layers,
        edit_output=patch_rep,
    ) as td:
        outputs_exp = model(**inp)

    # We report softmax probabilities for the answers_t token predictions of interest.
    probs = torch.softmax(outputs_exp.logits[1:, -1, :], dim=1).mean(dim=0)[answers_t]

    # If tracing all layers, collect all activations together to return.
    if trace_layers is not None:
        all_traced = torch.stack(
            [untuple(td[layer].output).detach().cpu() for layer in trace_layers], dim=2
        )
        return probs, all_traced

    return probs


def decode_tokens(tokenizer, token_array):
    if hasattr(token_array, "shape") and len(token_array.shape) > 1:
        return [decode_tokens(tokenizer, row) for row in token_array]
    return [tokenizer.decode([t]) for t in token_array]


def write_to_tracking(output_tracking_fn, save_name):
    """
    Write the instance name that has been saved to the tracking file (for easier restart)
    """
    if not os.path.exists(output_tracking_fn):
        result_to_save = [save_name]
    else:
        with open(output_tracking_fn, "r") as f_tracking:
            result_to_save = json.load(f_tracking)
            if save_name in result_to_save:
                return
            result_to_save.append(save_name)
    # Write list to json
    with open(output_tracking_fn, "w") as f_tracking:
        json.dump(result_to_save, f_tracking)


def plot_trace_heatmap(result, instance_name=None, title=None, xlabel=None, save_result=False,
                       examine_type_step_kind_output_path=None, visualize=False, other_info=None):
    if save_result:
        with open(f"{examine_type_step_kind_output_path}.jsonl", "a") as f_out:
            save_result = result.copy()
            save_result["name"] = instance_name
            if other_info is not None:
                save_result.update(other_info)
            for k, v in save_result.items(): save_result[k] = v.detach().cpu().numpy().tolist() if type(v) == torch.Tensor else v
            f_out.write(json.dumps(save_result) + "\n")

            write_to_tracking(f"{examine_type_step_kind_output_path}_tracking.jsonl", instance_name)
    if visualize:
        differences = result["scores"]
        if type(differences) != torch.Tensor:
            differences = torch.tensor(differences)
        low_score = result["low_score"]
        answer = result["answer"]
        kind = (
            None
            if (not result["kind"] or result["kind"] == "None")
            else str(result["kind"])
        )
        window = result.get("window", 10)
        labels = list(result["input_tokens"])
        for i in range(*result["subject_range"]):
            labels[i] = labels[i] + "*"

        with plt.rc_context():
            fig, ax = plt.subplots(figsize=(6, 5), dpi=200)
            vmax = None
            if kind == "attn":
                vmax = 0.55
            elif kind == "mlp":
                vmax = 0.6
            h = ax.pcolor(
                differences,
                cmap={None: "Purples", "None": "Purples", "hidden_state": "Purples", "mlp": "Greens", "attn": "Reds"}[
                    kind
                ],
                vmin=low_score,
                vmax=vmax,
            )
            ax.invert_yaxis()
            ax.set_yticks([0.5 + i for i in range(len(differences))])
            ax.set_xticks([0.5 + i for i in range(0, differences.shape[1], 5)])
            ax.set_xticklabels(list(range(0, differences.shape[1], 5)))
            ax.set_yticklabels(labels)
            kindname = "MLP" if kind == "mlp" else "Attn"
            ax.set_title(f"Impact of restoring {kindname} after corrupted input", fontsize=12)
            ax.set_xlabel(f"center of interval of {window} restored {kindname} layers")
            cb = plt.colorbar(h)
            # color bar display two decimal places
            cb.ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
            if title is not None:
                ax.set_title(title)
            if xlabel is not None:
                ax.set_xlabel(xlabel)
            elif answer is not None:
                # The following should be cb.ax.set_xlabel, but this is broken in matplotlib 3.5.1.
                cb.ax.set_title(str(answer).strip(), y=-0.16, fontsize=10)
            if instance_name:
                os.makedirs(os.path.dirname(instance_name), exist_ok=True)
                plt.savefig(f"{instance_name}.png", bbox_inches="tight")
                plt.close()
            else:
                plt.show()


def collect_embedding_std(mt, subjects):
    alldata = []
    for s in subjects:
        inp = make_inputs(mt.tokenizer, [s])
        with nethook.Trace(mt.model, layername(mt.model, 0, "embed")) as t:
            mt.model(**inp)
            alldata.append(t.output[0])
    alldata = torch.cat(alldata)
    noise_level = alldata.std().item()
    return noise_level


def merge_causal_tracing_figures(subfigure_fn_list, title, output_fn):
    # Put even number idx figures on the first row, odd number idx figures on the second row
    figure_num = len(subfigure_fn_list)
    nrows = 2
    ncols = figure_num // nrows
    image_obj_list = [Image.open(fn) for fn in subfigure_fn_list]
    image_size_list = [image_obj.size for image_obj in image_obj_list]
    fig_size = (sum([size[0] for size in image_size_list[ncols:]])/100, max([size[1] for size in image_size_list[ncols:]])/100 * nrows)
    fig, axes = plt.subplots(nrows, ncols, figsize=fig_size)
    for i, ax in zip([0, 2, 4, 1, 3, 5], axes.flat):
        ax.imshow(image_obj_list[i])
        ax.axis('off')

    fig.suptitle(title, fontsize=30, y=0.98)
    fig.tight_layout(pad=0)  # Reduce padding between images
    fig.subplots_adjust(top=0.95)  # Leave space for the title
    output_dir = os.path.dirname(output_fn)
    os.makedirs(output_dir, exist_ok=True)
    fig.savefig(output_fn)


def find_token_range(tokenizer, encoded_toks, target_answer_word):
    target_answer_word = target_answer_word.strip()
    decoded_toks = [tokenizer.decode([tok]) for tok in encoded_toks]
    for idx in range(len(decoded_toks)):
        if decoded_toks[idx].strip() == "" or target_answer_word.startswith(decoded_toks[idx].strip()) is False:
            continue
        for jdx in range(idx + 1, len(decoded_toks) + 1):
            tmp_parsed_str = tokenizer.decode(encoded_toks[idx:jdx]).strip()
            if tmp_parsed_str == target_answer_word:
                return (idx, jdx)
    return (-1, -1) # Not found