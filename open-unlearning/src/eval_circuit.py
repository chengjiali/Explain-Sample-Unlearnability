import os
import re
import json
# import wandb
import hydra
import torch
import torch.nn as nn
from torch import Tensor
import pandas as pd
from tqdm import tqdm
from functools import partial
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM
from transformer_lens import HookedTransformer
from einops import einsum
from eap.evaluate import evaluate_graph, evaluate_baseline
from eap.attribute import attribute
from eap.utils import tokenize_plus, make_hooks_and_matrices, compute_mean_activations
from eap.graph import Graph, AttentionNode

from data.qa import QADataset
from data.pretraining import PretrainingDataset, CompletionDataset
from data.collators import DataCollatorForSupervisedDataset
from evals.metrics.utils import evaluate_probability, eval_text_similarity, tokenwise_vocab_logprobs
from rouge_score import rouge_scorer

from trainer.utils import seed_everything
from model import get_model
from evals import get_evaluators

from typing import Callable, List, Union, Literal, Optional



def wandb_setup(cfg):
    parts = cfg.get('task_name').strip().split('/')
    mc_type = parts[0]
    mc_config = parts[1]
    mc_curve_type = parts[2]
    project = 'Mode Connectivity in Unlearning'
    group = f'eval_curve_{mc_type}_{mc_config}_{mc_curve_type}'

    if 'method' in mc_config:
        data, model, split, unlearn_method1, unlearn_method2 = parts[-1].split('_')

        name = [mc_type, mc_config, mc_curve_type, data, split, model, unlearn_method1, unlearn_method2]
        tags = [f"Experiment=eval_curve", 
                f"MC_type={mc_type}", f"MC_config={mc_config}", f"Curve_Type={mc_curve_type}", 
                f"Data={data}", f"Split={split}", f"Model={model}", 
                f"Unlearn_Method1={unlearn_method1}", f"Unlearn_Method2={unlearn_method2}"]

    else:
        data, model, split, unlearn_method = parts[-1].split('_')

        name = [mc_type, mc_config, mc_curve_type, data, split, model, unlearn_method]
        tags = [f"Experiment=eval_curve", 
                f"MC_type={mc_type}", f"MC_config={mc_config}", f"Curve_Type={mc_curve_type}", 
                f"Data={data}", f"Split={split}", f"Model={model}", 
                f"Unlearn_Method={unlearn_method}"]

    name = '-'.join(name)
    run_id = name
    # if 'method' not in mc_config and unlearn_method == 'GradAscent' and data == 'muse':
    #     run_id += '2'
    wandb.init(project=project, group=group, name=name, 
               config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True), 
               id=run_id, tags=tags, resume='allow')


@torch.inference_mode()
def evaluate_graph(model: HookedTransformer, graph: Graph, dataloader: DataLoader, 
                   metrics: Union[Callable[[Tensor],Tensor], List[Callable[[Tensor], Tensor]]], 
                   intervention: Literal['patching', 'zero', 'mean','mean-positional']='patching',  
                   ablate_non_circuit_edges=True, g2=None,
                   quiet=False, intervention_dataloader: Optional[DataLoader]=None, skip_clean:bool=True,
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
    """Evaluate a circuit (i.e. a graph where only some nodes are false, probably created by calling graph.apply_threshold). You probably want to prune 
        beforehand to make sure your circuit is valid.

    Args:
        model (HookedTransformer): The model to run the circuit on 
        graph (Graph): The circuit to evaluate
        dataloader (DataLoader): The dataset to evaluate on
        metrics (Union[Callable[[Tensor],Tensor], List[Callable[[Tensor], Tensor]]]): The metric(s) to evaluate with respect to
        quiet (bool, optional): Whether to silence the tqdm progress bar. Defaults to False.
        intervention (Literal[&#39;patching&#39;, &#39;zero&#39;, &#39;mean&#39;,&#39;mean, optional): Which ablation to evaluate with respect to. 
            'patching' is an interchange intervention; mean-positional takes the positional mean over the given dataset. Defaults to 'patching'.
        intervention_dataloader (Optional[DataLoader], optional): The dataset to take the mean over. Must be set if intervention is mean or mean-positional. Defaults to None.

    Returns:
        Union[torch.Tensor, List[torch.Tensor]]: A tensor (or list thereof) of faithfulness scores; if a list, each list entry 
            corresponds to a metric in the input list
    """
    assert model.cfg.use_attn_result, "Model must be configured to use attention result (model.cfg.use_attn_result)"
    if model.cfg.n_key_value_heads is not None:
        assert model.cfg.ungroup_grouped_query_attention, "Model must be configured to ungroup grouped attention (model.cfg.ungroup_grouped_attention)"
        
    assert intervention in ['patching', 'zero', 'mean', 'mean-positional'], f"Invalid intervention: {intervention}"
    
    if 'mean' in intervention:
        assert intervention_dataloader is not None, "Intervention dataloader must be provided for mean interventions"
        per_position = 'positional' in intervention
        means = compute_mean_activations(model, graph, intervention_dataloader, per_position=per_position)
        means = means.unsqueeze(0)
        if not per_position:
            means = means.unsqueeze(0)

    # This step cleans up the graph, removing components until it's fully connected
    graph.prune()

    # Construct a matrix that indicates which edges are in the graph
    in_graph_matrix = graph.in_graph.to(dtype=model.cfg.dtype)

    # Only zero out the unique edges compared to grpah2
    if g2 is not None:
        in_graph_matrix_g2 = g2.in_graph.to(dtype=model.cfg.dtype)
        unique_to_tensor1 = (in_graph_matrix == 1) & (in_graph_matrix_g2 == 0)
        in_graph_matrix[unique_to_tensor1] = 0

    
    in_graph_matrix = in_graph_matrix.to(device=model.cfg.device, dtype=model.cfg.dtype)
    
    # same thing but for neurons
    if graph.neurons_in_graph is not None:
        neuron_matrix = graph.neurons_in_graph.to(device=model.cfg.device, dtype=model.cfg.dtype)

        # If an edge is in the graph, but not all its neurons are, we need to update that edge anyway
        node_fully_in_graph = (neuron_matrix.sum(-1) == model.cfg.d_model).to(model.cfg.dtype)
        in_graph_matrix = einsum(in_graph_matrix, node_fully_in_graph, 'forward backward, forward -> forward backward')
    else:
        neuron_matrix = None

    # We take the opposite matrix, because we'll use it as a mask to specify 
    # which edges we want to corrupt
    if ablate_non_circuit_edges:
        print('Ablate non-circuit edges')
        in_graph_matrix = 1 - in_graph_matrix
        if neuron_matrix is not None:
            neuron_matrix = 1 - neuron_matrix
    else:
        print('Ablate circuit edges')
        
    if model.cfg.use_normalization_before_and_after:
        # If the model also normalizes the outputs of attention heads, we'll need to take that into account when evaluating the graph.
        attention_head_mask = torch.zeros((graph.n_forward, model.cfg.n_layers), device='cuda', dtype=model.cfg.dtype)
        for node in graph.nodes.values():
            if isinstance(node, AttentionNode):
                attention_head_mask[graph.forward_index(node), node.layer] = 1

        non_attention_head_mask = 1 - attention_head_mask.any(-1).to(dtype=model.cfg.dtype)
        attention_biases = torch.stack([block.attn.b_O for block in model.blocks])


    # For each node in the graph, corrupt its inputs, if the corresponding edge isn't in the graph 
    # We corrupt it by adding in the activation difference (b/w clean and corrupted acts)
    def make_input_construction_hook(activation_matrix, in_graph_vector, neuron_matrix):
        def input_construction_hook(activations, hook):
            # Case where layernorm is applied after attention (gemma only)
            if model.cfg.use_normalization_before_and_after:
                activation_differences = activation_matrix[0] - activation_matrix[1]
                
                # get the clean outputs of the attention heads that came before
                clean_attention_results = einsum(activation_matrix[1, :, :, :len(in_graph_vector)], 
                                                 attention_head_mask[:len(in_graph_vector)], 
                                                 'batch pos previous hidden, previous layer -> batch pos layer hidden')
                
                # get the update corresponding to non-attention heads, and the difference between clean and corrupted attention heads
                if neuron_matrix is not None:
                    non_attention_update = einsum(activation_differences[:, :, :len(in_graph_vector)], 
                                                  neuron_matrix[:len(in_graph_vector)], 
                                                  in_graph_vector, 
                                                  non_attention_head_mask[:len(in_graph_vector)], 
                                                  'batch pos previous hidden, previous hidden, previous ..., previous -> batch pos ... hidden')
                    corrupted_attention_difference = einsum(activation_differences[:, :, :len(in_graph_vector)], 
                                                            neuron_matrix[:len(in_graph_vector)], 
                                                            in_graph_vector, 
                                                            attention_head_mask[:len(in_graph_vector)], 
                                                            'batch pos previous hidden, previous hidden, previous ..., previous layer -> batch pos ... layer hidden')                    
                else:
                    non_attention_update = einsum(activation_differences[:, :, :len(in_graph_vector)], 
                                                  in_graph_vector, 
                                                  non_attention_head_mask[:len(in_graph_vector)], 
                                                  'batch pos previous hidden, previous ..., previous -> batch pos ... hidden')
                    corrupted_attention_difference = einsum(activation_differences[:, :, :len(in_graph_vector)], 
                                                            in_graph_vector, 
                                                            attention_head_mask[:len(in_graph_vector)], 
                                                            'batch pos previous hidden, previous ..., previous layer -> batch pos ... layer hidden')
                
                # add the biases to the attention results, and compute the corrupted attention results using the difference
                # we process all the attention heads at once; this is how we can tell if we're doing that
                if in_graph_vector.ndim == 2:
                    corrupted_attention_results = clean_attention_results.unsqueeze(2) + corrupted_attention_difference
                    # (1, 1, 1, layer, hidden)
                    clean_attention_results += attention_biases.unsqueeze(0).unsqueeze(0)
                    corrupted_attention_results += attention_biases.unsqueeze(0).unsqueeze(0).unsqueeze(0)
                else:
                    corrupted_attention_results = clean_attention_results + corrupted_attention_difference
                    clean_attention_results += attention_biases.unsqueeze(0).unsqueeze(0)
                    corrupted_attention_results += attention_biases.unsqueeze(0).unsqueeze(0)
                
                # pass both the clean and corrupted attention results through the layernorm and 
                # add the difference to the update
                update = non_attention_update
                valid_layers = attention_head_mask[:len(in_graph_vector)].any(0)
                for i, valid_layer in enumerate(valid_layers):
                    if not valid_layer:
                        break
                    if in_graph_vector.ndim == 2:
                        update -= model.blocks[i].ln1_post(clean_attention_results[:, :, None, i])
                        update += model.blocks[i].ln1_post(corrupted_attention_results[:, :, :, i])                        
                    else:
                        update -= model.blocks[i].ln1_post(clean_attention_results[:, :, i])
                        update += model.blocks[i].ln1_post(corrupted_attention_results[:, :, i])
                        
            else:
                # In the non-gemma case, things are easy!
                activation_differences = activation_matrix
                # The ... here is to account for a potential head dimension, when constructing a whole attention layer's input
                if neuron_matrix is not None:
                    update = einsum(activation_differences[:, :, :len(in_graph_vector)], neuron_matrix[:len(in_graph_vector)], in_graph_vector,
                                    'batch pos previous hidden, previous hidden, previous ... -> batch pos ... hidden')
                else:
                    update = einsum(activation_differences[:, :, :len(in_graph_vector)], in_graph_vector,
                                    'batch pos previous hidden, previous ... -> batch pos ... hidden')
            activations += update
            return activations
        return input_construction_hook

    def make_input_construction_hooks(activation_differences, in_graph_matrix, neuron_matrix):
        input_construction_hooks = []
        for layer in range(model.cfg.n_layers):
            # If any attention node in the layer is in the graph, just construct the input for the entire layer
            if any(graph.nodes[f'a{layer}.h{head}'].in_graph for head in range(model.cfg.n_heads)) and \
                not (neuron_matrix is None and all(parent_edge.in_graph for head in range(model.cfg.n_heads) for parent_edge in graph.nodes[f'a{layer}.h{head}'].parent_edges)):
                for i, letter in enumerate('qkv'):
                    node = graph.nodes[f'a{layer}.h0']
                    prev_index = graph.prev_index(node)
                    bwd_index = graph.backward_index(node, qkv=letter, attn_slice=True)
                    input_cons_hook = make_input_construction_hook(activation_differences, in_graph_matrix[:prev_index, bwd_index], neuron_matrix)
                    input_construction_hooks.append((node.qkv_inputs[i], input_cons_hook))
                    
            # add MLP hook if MLP in graph
            if graph.nodes[f'm{layer}'].in_graph and \
                not (neuron_matrix is None and all(parent_edge.in_graph for parent_edge in graph.nodes[f'm{layer}'].parent_edges)):
                node = graph.nodes[f'm{layer}']
                prev_index = graph.prev_index(node)
                bwd_index = graph.backward_index(node)
                input_cons_hook = make_input_construction_hook(activation_differences, in_graph_matrix[:prev_index, bwd_index], neuron_matrix)
                input_construction_hooks.append((node.in_hook, input_cons_hook))
                    
        # Always add the logits hook
        if not (neuron_matrix is None and all(parent_edge.in_graph for parent_edge in graph.nodes['logits'].parent_edges)):
            node = graph.nodes['logits']
            fwd_index = graph.prev_index(node)
            bwd_index = graph.backward_index(node)
            input_cons_hook = make_input_construction_hook(activation_differences, in_graph_matrix[:fwd_index, bwd_index], neuron_matrix)
            input_construction_hooks.append((node.in_hook, input_cons_hook))

        return input_construction_hooks
    
    # convert metrics to list if it's not already
    # if not isinstance(metrics, list):
    #     metrics = [metrics]
    results = {metric_name: [] for metric_name in metrics.keys()}
    
    # and here we actually run / evaluate the model
    dataloader = dataloader if quiet else tqdm(dataloader, desc='Data')
    for batch in dataloader:
        batch = {k: v.to('cuda') for k, v in batch.items()}
        batch_size = len(batch['input_ids'])
        n_pos = batch['attention_mask'].size(1)

        # fwd_hooks_corrupted adds in corrupted acts to activation_difference
        # fwd_hooks_clean subtracts out clean acts from activation_difference
        # activation difference is of size (batch, pos, src_nodes, hidden)
        (fwd_hooks_corrupted, fwd_hooks_clean, _), activation_difference = make_hooks_and_matrices(model, graph, batch_size, n_pos, None)
        
        input_construction_hooks = make_input_construction_hooks(activation_difference, in_graph_matrix, neuron_matrix)
        with torch.inference_mode():
            if intervention == 'patching':
                # We intervene by subtracting out clean and adding in corrupted activations
                with model.hooks(fwd_hooks_corrupted):
                    corrupted_logits = model(corrupted_tokens, attention_mask=attention_mask)
            else:
                # In the case of zero or mean ablation, we skip the adding in corrupted activations
                # but in mean ablations, we need to add the mean in
                if 'mean' in intervention:
                    activation_difference += means

            # For some metrics (e.g. accuracy or KL), we need the clean logits
            clean_logits = None if skip_clean else model(clean_tokens, attention_mask=attention_mask)

            for metric_name, metric_fn in metrics.items():
                if metric_name == 'rouge':
                    r = metric_fn(model, batch)
                else:
                    with model.hooks(fwd_hooks_clean + input_construction_hooks):
                        logits = model(batch['input_ids'], attention_mask=batch['attention_mask'])
                    r = metric_fn(logits, batch)
                if len(r.size()) == 0:
                    r = r.unsqueeze(0)
                results[metric_name].append(r)

    results = {metric_name: torch.cat(rs).cpu().float().numpy() for metric_name, rs in results.items()}
    return results
    # results = [torch.cat(rs) for rs in results]
    # unwrap the results if there's only one metric
    # if len(results) == 1:
    #     results = results[0]
    # return results.tolist()

@torch.inference_mode()
def evaluate_standard(model, dataloader, metrics, quiet=False):
    results = {metric_name: [] for metric_name in metrics.keys()}
    # and here we actually run / evaluate the model
    dataloader = dataloader if quiet else tqdm(dataloader)
    for batch in dataloader:
        batch = {k: v.to('cuda') for k, v in batch.items()}

        for metric_name, metric_fn in metrics.items():
            if metric_name == 'rouge':
                r = metric_fn(model, batch)
            else:
                logits = model(batch['input_ids'], attention_mask=batch['attention_mask'])
                r = metric_fn(logits, batch)
            if len(r.size()) == 0:
                r = r.unsqueeze(0)
            results[metric_name].append(r)

    results = {metric_name: torch.cat(rs).cpu().float().numpy() for metric_name, rs in results.items()}
    return results


IGNORE_INDEX = -100

def tokenwise_vocab_logprobs(logits, batch, return_labels=False):
    """Get vocabulary-wise log probabilities for each token in the sequence.

    Returns:
        log_probs_batch (List[Tensor]): List of tensors of shape (N, V) containing log probabilities
        for each sequence, where N is the length of labeled tokens and V is vocab size.
        labels_batch (List[Tensor]): List of tensors of length N. Returned only if return_labels is True
    """
    bsz, seq_len, V = logits.shape
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)[
        :, :-1, :
    ]  # Don't predict for last token

    # Process each sequence in batch separately
    log_probs_batch = []
    labels_batch = []
    for i in range(bsz):
        labels = batch["labels"][i]
        # Only include positions that have labels
        actual_indices = (labels != IGNORE_INDEX).nonzero(as_tuple=True)[0][
            :-1
        ]  # -1 to ignore eos prediction
        if len(actual_indices) == 0:
            labels_batch.append(torch.tensor([], device=labels.device))
            log_probs_batch.append(torch.zeros(0, V, device=labels.device))
            continue
        start_idx, end_idx = actual_indices[0].item(), actual_indices[-1].item()
        if start_idx == 0:
            print(
                "Index 0 in a datapoint's input_ids must not have loss (unignored labels) on it",
                UserWarning,
            )
        # Return full distribution for each position: shape (N, V)
        log_probs_batch.append(log_probs[i, start_idx - 1 : end_idx])
        labels_batch.append(labels[actual_indices])

    return (log_probs_batch, labels_batch) if return_labels else log_probs_batch


def parse_idx_values(raw, value_key=None):
    """
    Parse a dict like: {'0.0': {'score': 0.3}, '1.0': {'acc': 0.7}, ...}
    into a list of (idx: int, value: float).

    Parameters
    ----------
    raw : dict
        Outer keys are (stringified) indices, like "0.0", "1.0", etc.
        Inner values are dicts, e.g. {"score": 0.3} or {"acc": 0.7}.
    value_key : str or None
        If provided, use this inner key (e.g. "score", "acc").
        If None, infer it from the first inner dict.
    """
    # Auto-detect value_key if not given
    if value_key is None:
        for v in raw.values():
            if isinstance(v, dict) and len(v) > 0:
                value_key = list(v.keys())[-1]
                break
        if value_key is None:
            raise ValueError("Could not infer value key from inner dicts.")

    pairs = []

    for k, v in raw.items():
        # Convert index key like "0.0", "1", "12.0" -> int
        idx = int(float(k))

        if value_key not in v:
            raise KeyError(f"Inner key '{value_key}' not found in {v}")

        if isinstance(v[value_key], list):
            val = np.mean(v[value_key])
        else:
            val = float(v[value_key])
        pairs.append((idx, val))

    # Sort by idx
    pairs.sort(key=lambda x: x[0])
    return pairs


@hydra.main(version_base=None, config_path="../configs", config_name="eval.yaml")
def main(cfg: DictConfig):
    """Entry point of the code to evaluate models
    Args:
        cfg (DictConfig): Config to train
    """
    # wandb_setup(cfg)
    seed_everything(cfg.seed)
    model_cfg = cfg.model
    template_args = model_cfg.template_args
    assert model_cfg is not None, "Invalid model yaml passed in train config."
    hf_model, tokenizer = get_model(model_cfg)

    eval_cfgs = cfg.eval
    evaluators = get_evaluators(eval_cfgs)
    for evaluator_name, evaluator in evaluators.items():
        eval_args = {
            "template_args": template_args,
            "tokenizer": tokenizer,
        }

    # Parse MC configs
    task_name = cfg.get('task_name').strip()
    if 'phi-1_5' in task_name:
        task_name = task_name.replace('phi-1_5', 'phi-1.5')
    data_name, model_name, split, unlearn_method, hardness = task_name.split('_')


    def lm_loss(logits, batch):
        labels = batch["labels"]
        shifted_labels = labels[..., 1:].contiguous()
        logits = logits[..., :-1, :].contiguous()
        loss_function = nn.CrossEntropyLoss(ignore_index=-100, reduction="none")
        # agg loss across tokens
        losses = loss_function(logits.transpose(-1, -2), shifted_labels).sum(dim=-1)
        num_token_gt = (batch["labels"] != -100).sum(-1)
        avg_losses = losses / num_token_gt
        return avg_losses

    def rougeL(model, batch):
        generation_args = OmegaConf.create(
            {'do_sample': False, 'top_p': None, 'temperature': None, 'max_new_tokens': 32, 'use_cache': True,})
        
        def eval_rouge_recall_batch(gen_outputs, ground_truths):
            scorer = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)
            evals = []
            for gen, gt in zip(gen_outputs, ground_truths):
                rouge_scores = scorer.score(gt, gen)
                evals.append(
                    {
                        "rouge1_recall": rouge_scores["rouge1"].recall,
                        "rougeL_f1": rouge_scores["rougeL"].fmeasure,
                        "rougeL_recall": rouge_scores["rougeL"].recall,
                    }
                )
            return evals

        batch = {k: v.to(model.W_E.device) for k, v in batch.items()}
        input_ids = batch["input_ids"]
        labels = batch["labels"]
        input_texts = tokenizer.batch_decode(
            input_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        tokens = [label[label != IGNORE_INDEX] for label in labels]
        full_texts = tokenizer.batch_decode(
            tokens, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        ground_truths = [
            full_text.replace(input_text, "").strip()
            for input_text, full_text in zip(input_texts, full_texts)
        ]

        # convert to a simple dict from DictConfig
        generation_args = OmegaConf.to_container(generation_args, resolve=True)
        generation_args['use_past_kv_cache'] = False # generation_args.pop('use_cache')
        generation_args['verbose'] = False
        del generation_args['use_cache']
        output = model.generate(
            input_ids,
            **generation_args,
        )
        gen_texts = tokenizer.batch_decode(
            output[:, input_ids.shape[-1] :],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )

        stopwords = [tokenizer.decode([tokenizer.eos_token_id])]
        for i in range(len(gen_texts)):
            raw_text = gen_texts[i]
            for word in stopwords:
                if word and word in raw_text:
                    raw_text = raw_text.split(word)[0]
            raw_text = raw_text.strip()
            gen_texts[i] = raw_text

        scores = eval_rouge_recall_batch(gen_texts, ground_truths)
        scores = [
            {
                **rouge_evals,
                "input": input_text,
                "ground_truth": ground_truth,
                "generation": gen_text,
            }
            for rouge_evals, input_text, ground_truth, gen_text in zip(
                scores, input_texts, ground_truths, gen_texts
            )
        ]
        return torch.tensor([i['rougeL_recall'] for i in scores])

    def exact_mem(logits, batch):
        log_probs_batch, labels_batch = tokenwise_vocab_logprobs(logits, batch, return_labels=True)
        em_batch = []
        for log_probs, labels in zip(log_probs_batch, labels_batch):
            valid_len = len(labels)
            if valid_len == 0:
                # Rarely, tokenization can result in a mismatch with no valid target
                # tokens for loss computation (see preprocess_chat_instance() for
                # reference). Since this condition makes no sense in terms of
                # computing EM, we just choose to set EM=None
                print(
                    "EM score for an instance is marked None, due to "
                    "tokenization issues that resulted in no valid target tokens."
                )
                # em_batch.append({"score": None})
                em_batch.append({"score": 0})
            else:
                preds = torch.argmax(log_probs, dim=-1)
                em_score = (preds == labels).sum() / valid_len
                em_batch.append({"score": em_score.item()})

        return torch.tensor([i['score'] for i in em_batch])

    def extraction_strength(logits, batch):
        log_probs_batch, labels_batch = tokenwise_vocab_logprobs(logits, batch, return_labels=True)
        es_batch = []
        for log_probs, labels in zip(log_probs_batch, labels_batch):
            valid_len = len(labels)
            preds = torch.argmax(log_probs, dim=-1)
            for k in range(valid_len):
                suff_preds = preds[k:]
                suff_labels = labels[k:]
                if torch.equal(suff_preds, suff_labels):
                    break
            if valid_len == 0:
                # Rarely, tokenization can result in a mismatch with no valid target
                # tokens for loss computation (see preprocess_chat_instance() for
                # reference). Since this condition makes no sense in terms of
                # computing ES, we just choose to set ES=None
                print(
                    "ES score for an instance is marked None, due to "
                    "tokenization issues that resulted in no valid target tokens."
                )
                es_batch.append({"score": 0})
            else:
                es_score = 1 - (k / valid_len)
                es_batch.append({"score": es_score})
        
        return torch.tensor([i['score'] for i in es_batch])

    

    if model_name == 'Llama-3.2-1B-Instruct':
        tl_model_name = 'meta-llama/Llama-3.2-1B-Instruct'
    elif model_name == 'phi-1.5' or model_name == 'phi-1_5':
        tl_model_name = 'microsoft/phi-1_5'
    else:
        raise NotImplementedError

    # for hardness in ['easy', 'hard']:
    # Dataset
    if hardness == 'easy':
        with open(f'../sample_difficulty/{data_name}/{unlearn_method}_easy.txt') as f:
            subset_samples = f.read()
            subset_samples = [int(i) for i in subset_samples.split(',')]

    else:
        with open(f'../sample_difficulty/{data_name}/{unlearn_method}_hard.txt') as f:
            subset_samples = f.read()
            subset_samples = [int(i) for i in subset_samples.split(',')]

    print(f'Using {hardness.title()} samples', subset_samples)


    if data_name == 'tofu':
        hf_args = {"name": split, 'path': 'locuslab/TOFU', 'split': 'train'}
        dataset = QADataset(
            hf_args, template_args, tokenizer, max_length=512, predict_with_generate=False)
        collator = DataCollatorForSupervisedDataset(tokenizer, padding_side="right", index="index")

    elif data_name == 'muse':
        max_length = 2048
        hf_args = {"name": 'raw', 'path': f'muse-bench/MUSE-{split}', 'split': 'forget'}
        # dataset = CompletionDataset(
        #     hf_args, template_args, tokenizer, max_length=max_length, predict_with_generate=False, insert_space=True)
        dataset = PretrainingDataset(
            hf_args, template_args, tokenizer, max_length=max_length)
        collator = DataCollatorForSupervisedDataset(tokenizer, padding_side="left")

    elif data_name == 'wmdp':
        hf_args = {'path': 'text', 'data_files': f'data/wmdp/wmdp-corpora/{split}-forget-corpus.jsonl', 'split': 'train'}
        dataset = PretrainingDataset(
            hf_args, template_args, tokenizer, max_length=512)
        collator = DataCollatorForSupervisedDataset(tokenizer, padding_side="left",)

    # if data_name == 'tofu':
    #     dataset.data = dataset.data.select(subset_samples)
    # elif data_name == 'wmdp':
    #     dataset.chunks = [dataset.chunks[i] for i in subset_samples]

    data_loader = DataLoader(dataset, batch_size=2, shuffle=False, collate_fn=collator)

    topns = [50, 75, 100, 200]
    metrics = {metric_name: metric_fn for metric_name, metric_fn in zip(['loss', 'em', 'es', 'rouge'], [lm_loss, exact_mem, extraction_strength, rougeL])}


    # Find circuits on both the original model and the unlearned model
    for ori_or_unlearn in ['original', 'unlearn'][:1]:
        os.makedirs(f"saves/circuit/{ori_or_unlearn}", exist_ok=True)

        model = HookedTransformer.from_pretrained(
            tl_model_name,
            hf_model=hf_model if ori_or_unlearn == 'unlearn' else None,
            fold_ln=False,
            fold_value_biases=False,
            center_writing_weights=False,
            center_unembed=False,
            device='cuda',
            move_to_device=True,
            dtype=torch.bfloat16
        )
        # model.cfg.use_split_qkv_input = True
        model.cfg.use_attn_result = True
        # model.cfg.use_hook_mlp_in = True
        model.cfg.ungroup_grouped_query_attention = True
        model.eval()
        for n, p in model.named_parameters():
            p.requires_grad = False
            # print(p.requires_grad)

        
        for data_split in ['forget', 'retain'][:1]:
            if data_name == 'tofu':
                if data_split == 'forget':
                    hf_args = {"name": 'forget10', 'path': 'locuslab/TOFU', 'split': 'train'}
                    dataset = QADataset(
                        hf_args, template_args, tokenizer, max_length=512, predict_with_generate=False)
                    collator = DataCollatorForSupervisedDataset(tokenizer, padding_side="right", index="index")

                if data_split == 'retain':
                    hf_args = {"name": 'retain90', 'path': 'locuslab/TOFU', 'split': 'train'}
                    dataset = QADataset(
                        hf_args, template_args, tokenizer, max_length=512, predict_with_generate=False)
                    collator = DataCollatorForSupervisedDataset(tokenizer, padding_side="right", index="index")

            data_loader = DataLoader(dataset, batch_size=2, shuffle=False, collate_fn=collator)

            loss = pd.DataFrame({'idx': range(len(dataset)),})

            scores = evaluate_standard(model, data_loader, metrics)
            for metric_name, values in scores.items():
                loss[f'post_unlearn_{metric_name}'] = values


            # Evaluate circuit
            for topn in topns:

                # Evaluate all configs of edge ablation, E = All Edges
                # 1. Ablated edges = {e | e \in Easy_Circuit}
                # 2. Ablated edges = E - Case 1
                # 3. Ablated edges = {e | e \in Easy_Circuit, if e \notin Hard_Circuit}
                # 4. Ablated edges = E - Case 3
                i = 1
                for use_g2 in [False, True]:
                    for ablate_non_circuit_edges in [True, False]:
                        print(cfg.get('task_name'), ori_or_unlearn, hardness, data_split, 'Top N', topn, 'Use g2', use_g2, 'ablate_non_circuit_edges', ablate_non_circuit_edges)

                        g = Graph.from_json(f"saves/circuit/{ori_or_unlearn}/{cfg.get('task_name')}.json")
                        max_n = g.scores[g.scores != 0].numel()
                        if topn >= max_n:
                            continue
                        
                        g.apply_topn(int(topn))
                        if not os.path.exists(f"saves/circuit/{ori_or_unlearn}/{cfg.get('task_name')}_top{topn}.json"):
                            g.to_json(f"saves/circuit/{ori_or_unlearn}/{cfg.get('task_name')}_top{topn}.json")

                        if use_g2:
                            if hardness == 'easy':
                                g2 = Graph.from_json(f"saves/circuit/{ori_or_unlearn}/{cfg.get('task_name').replace(hardness, 'hard')}.json")
                            else:
                                g2 = Graph.from_json(f"saves/circuit/{ori_or_unlearn}/{cfg.get('task_name').replace(hardness, 'easy')}.json")

                            max_n = min(max_n, g2.scores[g2.scores != 0].numel())
                            if topn >= max_n:
                                continue

                            g2.apply_topn(int(topn))

                        else:
                            g2 = None

                        scores = evaluate_graph(
                            model, g, data_loader, metrics, intervention='zero', ablate_non_circuit_edges=ablate_non_circuit_edges, g2=g2)
                        for metric_name, values in scores.items():
                            loss[f'{metric_name}_topn_{topn}_{i}'] = values

                        i += 1


            for topn in topns:
                i = 1
                for use_g2 in [False, True]:
                    for ablate_non_circuit_edges in [True, False]:
                        print(cfg.get('task_name'), ori_or_unlearn, hardness, data_split, 'Greedy', topn, 'Use g2', use_g2, 'ablate_non_circuit_edges', ablate_non_circuit_edges)

                        g = Graph.from_json(f"saves/circuit/{ori_or_unlearn}/{cfg.get('task_name')}.json")
                        max_n = g.scores[g.scores != 0].numel()
                        if topn >= max_n:
                            continue
                        
                        g.apply_greedy(int(topn))
                        if not os.path.exists(f"saves/circuit/{ori_or_unlearn}/{cfg.get('task_name')}_greedy{topn}.json"):
                            g.to_json(f"saves/circuit/{ori_or_unlearn}/{cfg.get('task_name')}_greedy{topn}.json")

                        if use_g2:
                            if hardness == 'easy':
                                g2 = Graph.from_json(f"saves/circuit/{ori_or_unlearn}/{cfg.get('task_name').replace(hardness, 'hard')}.json")
                            else:
                                g2 = Graph.from_json(f"saves/circuit/{ori_or_unlearn}/{cfg.get('task_name').replace(hardness, 'easy')}.json")
                            
                            max_n = min(max_n, g2.scores[g2.scores != 0].numel())
                            if topn >= max_n:
                                continue
                            
                            g2.apply_greedy(int(topn))

                        else:
                            g2 = None
                        
                        scores = evaluate_graph(
                            model, g, data_loader, metrics, intervention='zero', ablate_non_circuit_edges=ablate_non_circuit_edges, g2=g2)
                        for metric_name, values in scores.items():
                            loss[f'{metric_name}_greedy_{topn}_{i}'] = values

            # with open(f"saves/circuit/{ori_or_unlearn}/{cfg.get('task_name')}_forget.json", 'w') as f:
                # json.dump(loss, f)
            # loss = pd.DataFrame(loss)
            loss.to_csv(f"saves/circuit/{ori_or_unlearn}/{cfg.get('task_name')}_{data_split}.csv", index=None)


    # for metric_name, metric_fn in zip(['loss', 'rouge', 'em', 'es'][1:], [lm_loss, rougeL, exact_mem, extraction_strength][1:]):
            # for subset in ['forget', 'retain']:
 
                # hf_args = {"name": subset, 'path': 'locuslab/TOFU', 'split': 'train'}
                # dataset = QADataset(
                #     hf_args, template_args, tokenizer, max_length=512, predict_with_generate=True if metric_name == 'rouge' else False)
                # collator = DataCollatorForSupervisedDataset(tokenizer, padding_side="right", index="index")
                # data_loader = DataLoader(dataset, batch_size=32, shuffle=False, collate_fn=collator)


        # eval_args['model'] = model
        # eval_args['overwrite'] = True
        # scores_standard = evaluator.evaluate(**eval_args)
        # res = {}
        # for metric_name, metric_value in scores_standard.items():
        #     if 'value_by_index' in metric_value:
        #         res[metric_name] = parse_idx_values(metric_value['value_by_index'])
        #     else:
        #         res[metric_name] = metric_value['agg_value']
        # loss[f'post_unlearn_{metric_name}'] = res

        #     # Evaluate entire model
        #     if metric_name == 'standard':
        #         eval_args['model'] = model
        #         # eval_args['']
        #         scores_standard = evaluator.evaluate(**eval_args)
        #         for _metric_name in ['forget_Q_A_ROUGE', 'retain_Q_A_ROUGE', 'ra_Q_A_ROUGE', 'wf_Q_A_ROUGE']:
        #             res = scores_standard[_metric_name]['value_by_index']
        #             res = [res[str(float(i))]['rougeL_recall'] for i in range(len(dataset))]
        #             loss[f'post_unlearn_{_metric_name}'] = res
        #     else:
        #         scores_standard = evaluate_standard(model, data_loader, metric_fn)
        #         loss[f'post_unlearn_{metric_name}'] = scores_standard

    # eval_cfgs = cfg.eval
    # evaluators = get_evaluators(eval_cfgs)
    # for evaluator_name, evaluator in evaluators.items():
    #     eval_args = {
    #         "template_args": template_args,
    #         "model": model,
    #         "tokenizer": tokenizer,
    #     }
    #     _ = evaluator.evaluate_mode_connectivity_curve(**eval_args)

    # Load model in torch.float32 for accurate loss
    # model_name = 'meta-llama/Llama-3.2-1B-Instruct'
    # repo = 'open-unlearning/unlearn_tofu_Llama-3.2-1B-Instruct_forget10_GradDiff_lr1e-05_alpha5_epoch10'

    # hf_model = AutoModelForCausalLM.from_pretrained(
    #     repo, 
    #     # low_cpu_mem_usage=True,
    #     torch_dtype=torch.float32)
    # hf_model = hf_model.to('cuda:0')
    # tl_model = HookedTransformer.from_pretrained(
    #     model_name,
    #     hf_model=hf_model,
    #     fold_ln=False,
    #     fold_value_biases=False,
    #     center_writing_weights=False,
    #     center_unembed=False,
    #     device='cuda:0',
    #     dtype=torch.float32
    # )

    # model.cfg.use_split_qkv_input = True
    # model.cfg.use_attn_result = True
    # model.cfg.use_hook_mlp_in = True
    # model.cfg.ungroup_grouped_query_attention = True

    # _ = tl_model.eval()
    # _ = hf_model.eval()

if __name__ == "__main__":
    main()
