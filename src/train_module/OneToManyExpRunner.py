import copy
import json
import os

import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers.utils import logging

from src.common_utils import STEP_TO_TOKEN_TYPE_NAME

logger = logging.get_logger(__name__)


class OneToManyExpRunner():
    def __init__(self, model, tokenizer, data_loader, args):
        self.model = model
        self.tokenizer = tokenizer
        self.data_loader = data_loader
        self.args = args

        self.target_answer_idx = int(self.args.target_answer_idx)
        self.layer_num = self.model.config.num_hidden_layers
        self.head_num = self.model.config.num_attention_heads
        self.output_dir = self.args.output_dir

    @torch.no_grad()
    def run_forward_pass_with_cache(self, input_batch, output_attentions=False, output_hidden_states=False):
        input_batch = {k: v.to(self.model.device) if type(v) == torch.Tensor else v for k, v in input_batch.items()}
        input_batch["output_attentions"] = output_attentions
        input_batch["output_hidden_states"] = output_hidden_states
        outputs = self.model.run_with_cache(**input_batch)
        return outputs

    @torch.no_grad()
    def get_actual_start_end_token_pos(self, target_token_start_end: tuple, target_last_token_position_from_right: int):
        """
        Get the actual start and end token position of the target token span in case if there is left padding
        :param target_token_start_end: (inclusive, exclusive). i.e. input_ids[target_token_start_end[0], target_token_start_end[1]] get the target token span
        :param target_last_token_position_from_right: inclusive. i.e. input_ids[target_last_token_position_from_right] get the last token of the target token span
        :return: (start_index, end_index)
        """
        if target_last_token_position_from_right is None:
            # None means current step's input does not contain the target token. e.g. answer_2 at step 1 or 2 (since answer 2 has not been generated yet)
            return (None, None)
        (b, e) = target_token_start_end if target_token_start_end is not None else (target_last_token_position_from_right, target_last_token_position_from_right + 1)
        if e - b == 1:
            start_index, end_index = target_last_token_position_from_right, target_last_token_position_from_right + 1
        else:
            end_index = target_last_token_position_from_right + 1  # +1 given originally end_index is an exclusive upper bound
            start_index = end_index - (e - b)

        end_index = None if end_index == 0 else end_index

        return start_index, end_index

    @torch.no_grad()
    def get_batch_knockout_start_end_idx_list(self, batch_target_start_end_idx, batch_right_to_target_token, input_batch_start_end_idx_list=None):
        """
        Get the actual start and end token position of all target token spans in a batch in case if there is left padding
        """
        # If input_batch_start_end_idx_list is None, it means the current step's input does not contain the target token. e.g. answer_2 at step 1 or 2 (since answer 2 has not been generated yet)
        batch_start_end_idx_list = input_batch_start_end_idx_list if input_batch_start_end_idx_list is not None else [[] for _ in range(len(batch_target_start_end_idx))]
        for batch_idx in range(len(batch_target_start_end_idx)):
            start_idx, end_idx = self.get_actual_start_end_token_pos(batch_target_start_end_idx[batch_idx], batch_right_to_target_token[batch_idx])
            batch_start_end_idx_list[batch_idx].append((start_idx, end_idx))
        return batch_start_end_idx_list

    def apply_layernorm(self, input_tensor, specified_outputs_for_layernorm, batch_entry_idx):
        """
        Apply LayerNorm to the input tensor using the final hidden states variance
        """
        original_dtype = input_tensor.dtype
        input_tensor = input_tensor.to(torch.float32)
        final_hidden_states_variance = specified_outputs_for_layernorm[1][f'model.norm.hook_final_hidden_states_variance'][batch_entry_idx][-1].to(self.model.device)
        input_tensor = input_tensor * torch.rsqrt(final_hidden_states_variance + self.model.model.norm.variance_epsilon)
        return self.model.model.norm.weight * input_tensor.to(original_dtype)

    def get_unembed_results(self, layer_normed_input_tensor):
        """
        Get the unembedded results of the input tensor
        """
        output_logits = self.model.lm_head(layer_normed_input_tensor)
        output_probs = F.softmax(output_logits, dim=-1)
        pred_tokens = torch.argmax(output_probs, dim=-1)
        return {
            "logits": output_logits,
            "probs": output_probs,
            "pred_tokens": pred_tokens,
        }

    @torch.no_grad()
    def do_decode_attention_mlp(self):
        self.model.eval()

        tmp_output_dir = f"{self.output_dir}"
        os.makedirs(tmp_output_dir, exist_ok=True)
        attn_out_fn = os.path.join(tmp_output_dir, f"attn_output.jsonl")
        mlp_out_fn = os.path.join(tmp_output_dir, f"mlp_output.jsonl")

        progress = tqdm(self.data_loader, total=len(self.data_loader))
        for batch in progress:
            input_batch = {'input_ids': batch['input_ids'], 'attention_mask': batch['attention_mask']}
            normal_outputs = self.run_forward_pass_with_cache(input_batch, output_hidden_states=True)

            for layer_idx in range(self.layer_num):
                for batch_entry_idx in range(self.data_loader.batch_size):
                    entry_three_answers_first_token_list = batch['three_answers_first_token_list'][batch_entry_idx]
                    tmp_subject_first_token_id = batch['subject_first_token'][batch_entry_idx]

                    def output_unembed_results(input_tensor, output_fn):
                        # Use the final hidden states variance to normalize the input tensor
                        layer_normed_input_tensor = self.apply_layernorm(input_tensor, normal_outputs, batch_entry_idx)
                        unembed_results = self.get_unembed_results(layer_normed_input_tensor)
                        output_logits = unembed_results["logits"]
                        output_probs = unembed_results["probs"]
                        pred_tokens = unembed_results["pred_tokens"]

                        result = {
                            # Basic info
                            "layer_idx": layer_idx,
                            "query": batch['query'][batch_entry_idx],
                            "target_answer_name": batch['target_answer_name'][batch_entry_idx],
                            "target_answer_tokens": batch['target_answer_tokens'][batch_entry_idx],
                            "three_answers_first_token_list": entry_three_answers_first_token_list,
                            "subject_first_token_id": tmp_subject_first_token_id,

                            # Results of the subject and three answer tokens
                            "subject_first_token": tmp_subject_first_token_id,
                            "subject_first_token_prob": output_probs[tmp_subject_first_token_id].item(),
                            "subject_first_token_logit": unembed_results["logits"][tmp_subject_first_token_id].item(),

                            "three_answer_first_token": entry_three_answers_first_token_list,
                            "three_answer_first_token_prob": [output_probs[token_idx].item() for token_idx in entry_three_answers_first_token_list],
                            "three_answer_first_token_logit": [unembed_results["logits"][token_idx].item() for token_idx in entry_three_answers_first_token_list],

                            # Comment back if you need them
                            # "pred_token": pred_tokens,
                            # "pred_token_str": self.tokenizer.decode(pred_tokens, skip_special_tokens=False),
                            # "pred_token_logit": unembed_results["logits"][pred_tokens].item(),
                            # "pred_token_prob": output_probs[pred_tokens].item(),
                        }
                        with open(output_fn, "a") as f_out:
                            f_out.write(json.dumps(result) + "\n")

                    # Attention
                    hooked_attention_outputs = normal_outputs[1][f'model.layers.{layer_idx}.hook_attn_output'][batch_entry_idx]
                    output_unembed_results(hooked_attention_outputs, attn_out_fn)

                    # MLP
                    hooked_MLP_outputs = normal_outputs[1][f'model.layers.{layer_idx}.hook_mlp_output'][batch_entry_idx]
                    output_unembed_results(hooked_MLP_outputs, mlp_out_fn)

    @torch.no_grad()
    def do_token_lens_and_attention_knockout(self):
        self.model.eval()
        # Prepare output files
        if self.args.do_token_lens:
            token_type_to_token_lens_output_f = {}
            for token_type in STEP_TO_TOKEN_TYPE_NAME[f"{self.target_answer_idx}"]:
                token_type_to_token_lens_output_f[token_type] = open(os.path.join(self.output_dir, f"token_lens_{token_type}_results.jsonl"), "a")
        if self.args.do_attention_knockout:
            token_type_to_knockout_output_f = {}
            for token_type in STEP_TO_TOKEN_TYPE_NAME[f"{self.target_answer_idx}"]:
                token_type_to_knockout_output_f[token_type] = open(os.path.join(self.output_dir, f"attn_knockout_{token_type}_results.jsonl"), "a")

        progress = tqdm(self.data_loader, total=len(self.data_loader))

        for batch in progress:
            normal_batch = {'input_ids': batch['input_ids'], 'attention_mask': batch['attention_mask']}
            normal_outputs = self.run_forward_pass_with_cache(normal_batch, output_attentions=True, output_hidden_states=True)
            normal_outputs_last_token_logits = normal_outputs[0].logits[:, -1, :]
            normal_outputs_probs = F.softmax(normal_outputs_last_token_logits, dim=-1)

            if self.args.do_attention_knockout:
                # Get outputs when knocking out specified tokens
                token_type_to_knockout_outputs = {}

                def get_specified_token_knockout_outputs(specified_token_name, specified_token_start_end_idx, specified_token_position_from_right):
                    tmp_batch = copy.deepcopy(normal_batch)
                    tmp_batch["batch_knockout_start_end_idx_list"] = self.get_batch_knockout_start_end_idx_list(specified_token_start_end_idx, specified_token_position_from_right)
                    token_type_to_knockout_outputs[specified_token_name] = self.run_forward_pass_with_cache(tmp_batch, output_attentions=True)

                get_specified_token_knockout_outputs("subject", batch['subject_start_end_idx'], batch['subject_last_token_position_from_right'])
                get_specified_token_knockout_outputs("last_token", batch['last_token_start_end_idx'], batch['last_token_position_from_right'])

                if self.target_answer_idx >= 2:
                    get_specified_token_knockout_outputs("answer_1", batch['prev_answer_1_start_end_idx'], batch['right_to_answer_1_last_token'])

                if self.target_answer_idx == 3:
                    get_specified_token_knockout_outputs("answer_2", batch['prev_answer_2_start_end_idx'], batch['right_to_answer_2_last_token'])

            # Do token lens and attention knockout logit process
            if self.args.do_token_lens or self.args.do_attention_knockout:
                for batch_entry_idx in range(self.data_loader.batch_size):
                    # Get the actual start and end position of the subject and three answers in case there is padding on the left
                    entry_three_answers_first_token_list = batch['three_answers_first_token_list'][batch_entry_idx]
                    tmp_subject_first_token_id = batch['subject_first_token'][batch_entry_idx]

                    tmp_subject_last_token_position_from_right = batch['subject_last_token_position_from_right'][batch_entry_idx]
                    tmp_subject_start_end_idx = batch['subject_start_end_idx'][batch_entry_idx]

                    tmp_right_to_answer_1_last_token = batch['right_to_answer_1_last_token'][batch_entry_idx] if self.target_answer_idx >= 2 else None
                    tmp_prev_answer_1_start_end_idx = batch['prev_answer_1_start_end_idx'][batch_entry_idx] if self.target_answer_idx >= 2 else None

                    tmp_right_to_answer_2_last_token = batch['right_to_answer_2_last_token'][batch_entry_idx] if self.target_answer_idx == 3 else None
                    tmp_prev_answer_2_start_end_idx = batch['prev_answer_2_start_end_idx'][batch_entry_idx] if self.target_answer_idx == 3 else None

                    token_type_to_actual_start_end_list = {
                        "subject": [self.get_actual_start_end_token_pos(tmp_subject_start_end_idx, tmp_subject_last_token_position_from_right)],
                        "answer_1": [self.get_actual_start_end_token_pos(tmp_prev_answer_1_start_end_idx, tmp_right_to_answer_1_last_token)],
                        "answer_2": [self.get_actual_start_end_token_pos(tmp_prev_answer_2_start_end_idx, tmp_right_to_answer_2_last_token)],
                        "last_token": [(-1, None)],
                    }

                    for layer_idx in range(self.layer_num):
                        def do_token_lens(specified_outputs, output_f, specified_start_end_idx_list: list):
                            # Get cached outputs from HookPoints for Attention Lens
                            layer_head_value_vectors = specified_outputs[1][f'model.layers.{layer_idx}.self_attn.hook_heads_value_vectors'].transpose(1, 2).contiguous()[batch_entry_idx]
                            concat_weighted_value_vector = None
                            for head_idx in range(self.head_num):
                                weighted_head_target_token_value_vector_sum = None
                                # Get all weighted value vectors of target token from the current head (based on the weight from the last to target tokens)
                                for tmp_target_start_end_idx in specified_start_end_idx_list:
                                    start_index, end_index = tmp_target_start_end_idx
                                    head_last_to_target_token_weight = specified_outputs[0].attentions[layer_idx][batch_entry_idx][head_idx][-1][start_index:end_index].detach().cpu()
                                    head_target_token_value_vector = layer_head_value_vectors[start_index:end_index, head_idx].detach().cpu()
                                    weighted_head_target_token_value_vector_sum = torch.sum(head_target_token_value_vector * head_last_to_target_token_weight.unsqueeze(1), dim=0) if weighted_head_target_token_value_vector_sum is None else weighted_head_target_token_value_vector_sum + torch.sum(head_target_token_value_vector * head_last_to_target_token_weight.unsqueeze(1), dim=0)
                                # Concatenate weighted value vectors from all heads in the current layer
                                concat_weighted_value_vector = weighted_head_target_token_value_vector_sum if concat_weighted_value_vector is None else torch.concatenate((concat_weighted_value_vector, weighted_head_target_token_value_vector_sum))
                            target_token_attention_output = concat_weighted_value_vector.to(self.model.device)
                            token_lens_output = self.model.model.layers[layer_idx].self_attn.o_proj(target_token_attention_output)

                            # LayerNorm the output using the last hidden states variance
                            token_lens_output = self.apply_layernorm(token_lens_output, specified_outputs, batch_entry_idx)
                            unembed_token_lens_results = self.get_unembed_results(token_lens_output)

                            token_lens_output_logits = unembed_token_lens_results["logits"]
                            token_lens_output_probs = unembed_token_lens_results["probs"]
                            token_lens_output_pred_tokens = unembed_token_lens_results["pred_tokens"]

                            tmp_output_result_dict = {
                                # Basic info
                                "query": batch['query'][batch_entry_idx],
                                "target_answer_idx": self.target_answer_idx,
                                "target_answer_name": batch['target_answer_name'][batch_entry_idx],
                                "target_answer_tokens": batch['target_answer_tokens'][batch_entry_idx],
                                "subject_first_token_id": tmp_subject_first_token_id,
                                "three_answers_first_token_list": entry_three_answers_first_token_list,
                                "layer_idx": layer_idx,

                                # Token lens results
                                "three_answer_first_token_logit": [token_lens_output_logits[token_idx].item() for token_idx in entry_three_answers_first_token_list],
                                "subject_first_token_logit": token_lens_output_logits[tmp_subject_first_token_id].item(),

                                # Comment back if you need them
                                # "attn_lens_output_pred_token": token_lens_output_pred_tokens.item(),
                                # "attn_lens_output_pred_token_logit": token_lens_output_logits[token_lens_output_pred_tokens].item(),
                                # "attn_lens_output_pred_token_str": self.tokenizer.decode(token_lens_output_pred_tokens.item(), skip_special_tokens=False),
                                # "attn_lens_output_pred_token_prob": token_lens_output_probs[token_lens_output_pred_tokens].item(),
                            }
                            output_f.write(json.dumps(tmp_output_result_dict) + "\n")

                        def do_attention_knockout(specified_full_outputs, specified_knockout_outputs, output_f):
                            tmp_output_result_dict = {
                                # Basic info
                                "query": batch['query'][batch_entry_idx],
                                "subject_first_token_id": tmp_subject_first_token_id,
                                "target_answer_idx": self.target_answer_idx,
                                "target_answer_name": batch['target_answer_name'][batch_entry_idx],
                                "target_answer_tokens": batch['target_answer_tokens'][batch_entry_idx],
                                "three_answers_first_token_list": entry_three_answers_first_token_list,
                                "layer_idx": layer_idx,

                                # Comment back if you need them
                                # "model_pred_token": normal_outputs_pred_tokens[batch_entry_idx].item(),
                                # "model_pred_token_str": self.tokenizer.decode(normal_outputs_pred_tokens[batch_entry_idx], skip_special_tokens=False),
                                # "model_pred_token_prob": normal_outputs_probs[batch_entry_idx][normal_outputs_pred_tokens[batch_entry_idx]].item(),
                                # "model_pred_token_logit": normal_outputs_last_token_logits[batch_entry_idx][normal_outputs_pred_tokens[batch_entry_idx]].item(),
                                # "target_answer_token_prob": normal_outputs_probs[batch_entry_idx][batch['target_answer_tokens][batch_entry_idx][0]].item(),
                                # "target_answer_token_logit": normal_outputs_last_token_logits[batch_entry_idx][batch['target_answer_tokens][batch_entry_idx][0]].item(),
                            }

                            full_output = specified_full_outputs[1][f'model.layers.{layer_idx}.hook_mlp_output'][batch_entry_idx].to(self.model.device)
                            attention_knockout_output = specified_knockout_outputs[1][f'model.layers.{layer_idx}.hook_mlp_output'][batch_entry_idx].to(self.model.device)
                            full_output_results = self.get_unembed_results(self.apply_layernorm(full_output, specified_full_outputs, batch_entry_idx))
                            attention_knockout_results = self.get_unembed_results(self.apply_layernorm(attention_knockout_output, specified_knockout_outputs, batch_entry_idx))

                            def update_result_dict_with_new_output_results(new_output_results, new_output_name):
                                tmp_output_result_dict.update({
                                    f"subject_first_token_{new_output_name}_logit": new_output_results["logits"][tmp_subject_first_token_id].item(),
                                    f"three_answer_first_token_{new_output_name}_logit": [new_output_results["logits"][token_idx].item() for token_idx in entry_three_answers_first_token_list],

                                    # Comment back if you need them
                                    # f"three_answer_first_token_{new_output_name}_prob": [new_output_results["probs"][token_idx].item() for token_idx in entry_three_answers_first_token_list],
                                    # f"subject_first_token_{new_output_name}_prob": new_output_results["probs"][tmp_subject_first_token_id].item(),
                                    # f"{new_output_name}_output_pred_tokens": new_output_results["pred_tokens"].item(),
                                    # f"{new_output_name}_output_pred_token_str": self.tokenizer.decode(new_output_results["pred_tokens"].item(), skip_special_tokens=False),
                                })
                            update_result_dict_with_new_output_results(full_output_results, f"full_mlp")
                            update_result_dict_with_new_output_results(attention_knockout_results,f"attn_knockout_mlp")
                            output_f.write(json.dumps(tmp_output_result_dict) + "\n")

                        if self.args.do_token_lens:
                            for token_type, target_start_end_idx_list in token_type_to_actual_start_end_list.items():
                                if target_start_end_idx_list[0][0] is not None:
                                    do_token_lens(normal_outputs, token_type_to_token_lens_output_f[token_type], target_start_end_idx_list)

                        if self.args.do_attention_knockout:
                            for token_type in STEP_TO_TOKEN_TYPE_NAME[f"{self.target_answer_idx}"]:
                                do_attention_knockout(normal_outputs, token_type_to_knockout_outputs[token_type], token_type_to_knockout_output_f[token_type])

        if self.args.do_token_lens:
            for out_f in token_type_to_token_lens_output_f.values():
                out_f.flush()
                out_f.close()
        if self.args.do_attention_knockout:
            for out_f in token_type_to_knockout_output_f.values():
                out_f.flush()
                out_f.close()

    @torch.no_grad()
    def examine_finegrained_attention_head(self):
        self.model.eval()
        d_head = self.model.config.hidden_size // self.head_num

        progress = tqdm(self.data_loader, total=len(self.data_loader))
        for batch in progress:
            normal_batch = {'input_ids': batch['input_ids'], 'attention_mask': batch['attention_mask']}
            normal_outputs = self.run_forward_pass_with_cache(normal_batch, output_attentions=True)
            normal_outputs_last_token_logits = normal_outputs[0].logits[:, -1, :]
            normal_outputs_probs = F.softmax(normal_outputs_last_token_logits, dim=-1)

            for batch_entry_idx in range(self.data_loader.batch_size):
                for layer_idx in range(self.layer_num):
                    entry_three_answers_first_token_list = batch['three_answers_first_token_list'][batch_entry_idx]
                    tmp_subject_first_token_id = batch['subject_first_token'][batch_entry_idx]

                    layer_head_last_to_target_token_weight = normal_outputs[0].attentions[layer_idx][batch_entry_idx, :, -1]
                    layer_head_value_vectors = normal_outputs[1][f'model.layers.{layer_idx}.self_attn.hook_heads_value_vectors'].contiguous()[batch_entry_idx]
                    layer_head_weighted_value_vector_sum = (layer_head_value_vectors * layer_head_last_to_target_token_weight.unsqueeze(-1)).sum(dim=1)
                    projected_layer_head_output = (layer_head_weighted_value_vector_sum.unsqueeze(-1) * self.model.model.layers[layer_idx].self_attn.o_proj.weight.view(self.head_num, d_head, -1)).sum(dim=1)

                    tmp_layer_output_jsonl_fn = os.path.join(self.output_dir, f"layer_{layer_idx}_output.jsonl")
                    tmp_layer_output_dir = {
                        "query": batch['query'][batch_entry_idx],
                        "target_answer_idx": self.target_answer_idx,
                        "target_answer_name": batch['target_answer_name'][batch_entry_idx],
                        "target_answer_tokens": batch['target_answer_tokens'][batch_entry_idx],
                        "subject_first_token_id": tmp_subject_first_token_id,
                        "three_answers_first_token_list": entry_three_answers_first_token_list,
                    }
                    output_f = open(tmp_layer_output_jsonl_fn, "a")

                    for head_idx in range(self.head_num):
                        tmp_head_output = projected_layer_head_output[head_idx]
                        layer_normed_input_tensor = self.apply_layernorm(tmp_head_output, normal_outputs, batch_entry_idx)
                        unembed_layer_normed_input_tensor = self.get_unembed_results(layer_normed_input_tensor)
                        output_logits = unembed_layer_normed_input_tensor["logits"]
                        output_probs = unembed_layer_normed_input_tensor["probs"]
                        pred_tokens = unembed_layer_normed_input_tensor["pred_tokens"]

                        tmp_layer_output_dir[f"head_{head_idx}"] = {
                            "subject_first_token_logit": output_logits[tmp_subject_first_token_id].item(),
                            "three_answer_first_token_logit": [output_logits[token_idx].item() for token_idx in entry_three_answers_first_token_list],

                            # Comment back if you need them
                            # "pred_token": pred_tokens,
                            # "pred_token_str": self.tokenizer.decode(pred_tokens, skip_special_tokens=False),
                            # "pred_token_logit": output_logits[pred_tokens].item(),
                        }
                    output_f.write(json.dumps(tmp_layer_output_dir) + "\n")
                    output_f.flush()
                    output_f.close()