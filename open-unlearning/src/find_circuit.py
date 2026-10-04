import os
import re
import wandb
import hydra
import torch
import torch.nn as nn
import pandas as pd
from tqdm import tqdm
from functools import partial
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM
from transformer_lens import HookedTransformer
from eap.graph import Graph
from eap.evaluate import evaluate_graph, evaluate_baseline
from eap.attribute import attribute
from eap.utils import make_hooks_and_matrices

from data.qa import QADataset
from data.pretraining import PretrainingDataset, CompletionDataset
from data.collators import DataCollatorForSupervisedDataset
from evals.metrics.utils import evaluate_probability, eval_text_similarity, tokenwise_vocab_logprobs

from trainer.utils import seed_everything
from model import get_model
from evals import get_evaluators


from typing import Callable, List, Union, Literal, Optional

import torch
from torch import Tensor
from torch.utils.data import DataLoader
from transformer_lens import HookedTransformer
from tqdm import tqdm
from einops import einsum

from eap.utils import tokenize_plus, make_hooks_and_matrices, compute_mean_activations
from eap.graph import Graph, AttentionNode


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

def get_scores_ig_activations(model, graph, dataloader, metric, steps=5):

    scores = torch.zeros((graph.n_forward, graph.n_backward), device='cuda', dtype=model.cfg.dtype)    
    
    total_items = 0
    for batch in tqdm(dataloader):
        batch = {k: v.to('cuda') for k, v in batch.items()}
        batch_size = len(batch['input_ids'])
        total_items += batch_size
        n_pos = batch['attention_mask'].size(1)

        # clean_tokens, attention_mask, input_lengths, n_pos = tokenize_plus(model, clean)
        # corrupted_tokens, _, _, _ = tokenize_plus(model, corrupted)

        (_, _, bwd_hooks), activation_difference = make_hooks_and_matrices(model, graph, batch_size, n_pos, scores)
        (fwd_hooks_corrupted, _, _), activations_corrupted = make_hooks_and_matrices(model, graph, batch_size, n_pos, scores)
        (fwd_hooks_clean, _, _), activations_clean = make_hooks_and_matrices(model, graph, batch_size, n_pos, scores)

        with torch.no_grad():
            with model.hooks(fwd_hooks=fwd_hooks_clean):
                clean_logits = model(batch['input_ids'], attention_mask=batch['attention_mask'])
                activation_difference += activations_corrupted.clone().detach() - activations_clean.clone().detach()

        # tqdm.write(f'{activation_difference.sum().item()}, {activations_clean.sum().item()}, {activations_corrupted.sum().item()}')
        def output_interpolation_hook(k: int, clean: torch.Tensor, corrupted: torch.Tensor):
            def hook_fn(activations: torch.Tensor, hook):
                alpha = k/steps
                new_output = alpha * clean + (1 - alpha) * corrupted
                return new_output
            return hook_fn

        total_steps = 0

        nodeslist = [graph.nodes['input']]
        for layer in range(graph.cfg['n_layers']):
            nodeslist.append(graph.nodes[f'a{layer}.h0'])
            nodeslist.append(graph.nodes[f'm{layer}'])

        for node in tqdm(nodeslist, desc='Nodelist', leave=False):
            for step in range(1, steps+1):
                total_steps += 1
                
                clean_acts = activations_clean[:, :, graph.forward_index(node)]
                corrupted_acts = activations_corrupted[:, :, graph.forward_index(node)]
                fwd_hooks = [(node.out_hook, output_interpolation_hook(step, clean_acts, corrupted_acts))]

                model.zero_grad()
                with model.hooks(fwd_hooks=fwd_hooks, bwd_hooks=bwd_hooks):
                    logits = model(batch['input_ids'], attention_mask=batch['attention_mask'])
                    metric_value = metric(clean_logits, logits, batch)
                    metric_value = metric_value.mean()
                    metric_value.backward()#retain_graph=True)

    scores /= total_items
    scores /= total_steps

    graph.scores[:] =  scores.to(graph.scores.device)

    return scores

@torch.inference_mode()
def evaluate_graph(model: HookedTransformer, graph: Graph, dataloader: DataLoader, 
                   metrics: Union[Callable[[Tensor],Tensor], List[Callable[[Tensor], Tensor]]], 
                   quiet=False, intervention: Literal['patching', 'zero', 'mean','mean-positional']='patching', 
                   intervention_dataloader: Optional[DataLoader]=None, skip_clean:bool=True, 
                   ablate_non_circuit_edges=True) -> Union[torch.Tensor, List[torch.Tensor]]:
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
    in_graph_matrix = graph.in_graph.to(device=model.cfg.device, dtype=model.cfg.dtype)
    
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
    if not isinstance(metrics, list):
        metrics = [metrics]
    results = [[] for _ in metrics]
    
    # and here we actually run / evaluate the model
    dataloader = dataloader if quiet else tqdm(dataloader)
    for batch in tqdm(dataloader):
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
                
            with model.hooks(fwd_hooks_clean + input_construction_hooks):
                logits = model(batch['input_ids'], attention_mask=batch['attention_mask'])

        for i, metric in enumerate(metrics):
            r = metric(logits, batch)
            if len(r.size()) == 0:
                r = r.unsqueeze(0)
            results[i].append(r)

    results = [torch.cat(rs) for rs in results]
    # unwrap the results if there's only one metric
    if len(results) == 1:
        results = results[0]
    return results.tolist()

@torch.inference_mode()
def evaluate_standard(model, dataloader, metric, quiet=False):
    results = []    
    # and here we actually run / evaluate the model
    dataloader = dataloader if quiet else tqdm(dataloader)
    for batch in tqdm(dataloader):
        batch = {k: v.to('cuda') for k, v in batch.items()}
        batch_size = len(batch['input_ids'])
        n_pos = batch['attention_mask'].size(1)

        logits = model(batch['input_ids'], attention_mask=batch['attention_mask'])

        r = metric(logits, batch)
        results.extend(r.tolist())

    return results

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

    # Parse MC configs
    task_name = cfg.get('task_name').strip()
    if 'phi-1_5' in task_name:
        task_name = task_name.replace('phi-1_5', 'phi-1.5')
    data_name, model_name, split, unlearn_method, hardness = task_name.split('_')

    # Dataset
    order = pd.read_csv('../sample_difficulty/tofu/loss.csv')
    order = order[order.method == unlearn_method]['order'].tolist()
    if hardness == 'easy':
        subset_samples = order[-50:]
    else:
        subset_samples = order[:50]

    print(f'Using {hardness.title()} samples', subset_samples)


    if data_name == 'tofu':
        hf_args = {"name": 'retain90', 'path': 'locuslab/TOFU', 'split': 'train'}
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

    if data_name == 'tofu':
    #     subset_samples = [ 758, 1380, 1398, 3513, 1097, 2348, 1222, 1873, 3295, 3160, 2564,
    #    1304,  897, 2930,  394, 1144, 1944, 1828, 3500, 1068, 3296,  778,
    #    2342, 2211, 1265, 1811,  502, 2605, 2296, 1840, 3312, 1677, 1407,
    #    2322, 1485,  432,  794,  651, 2547, 2964,  619,  789,   93, 2152,
    #    1716, 3398,  312, 2839,  682, 1008,  806,  222,  760, 2248, 1554,
    #    3106, 3557, 2208,  429, 1596, 3118, 3591, 2491, 3425, 1149,  485,
    #    1367, 2162, 1440, 2531,   17, 2249,  840,  873, 2776, 2002, 2262,
    #    1965, 3335, 2784, 2715, 3584,  473, 3449, 1929,  128, 3343, 1203,
    #    1134, 1835, 1698, 1060,  950,  210,  750, 2832, 1119,  146, 3535,
    #     756]
        dataset.data = dataset.data.select(subset_samples)
    elif data_name == 'wmdp':
        dataset.chunks = [dataset.chunks[i] for i in subset_samples]

    data_loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collator)

    def metric_fn(clean_logits, logits, batch):
        labels = batch["labels"]
        shifted_labels = labels[..., 1:].contiguous()
        logits = logits[..., :-1, :].contiguous()
        loss_function = nn.CrossEntropyLoss(ignore_index=-100, reduction="none")
        # agg loss across tokens
        losses = loss_function(logits.transpose(-1, -2), shifted_labels).sum(dim=-1)
        num_token_gt = (batch["labels"] != -100).sum(-1)
        avg_losses = losses / num_token_gt
        normalized_probs = torch.exp(-avg_losses)

        clean_logits = clean_logits[..., :-1, :].contiguous()
        # agg loss across tokens
        clean_losses = loss_function(clean_logits.transpose(-1, -2), shifted_labels).sum(dim=-1)
        clean_avg_losses = clean_losses / num_token_gt
        clean_normalized_probs = torch.exp(-clean_avg_losses)

        prob_diff = torch.nn.functional.kl_div(normalized_probs, clean_normalized_probs, log_target=True, reduction='none')
        prob_diff = prob_diff.mean()

        return prob_diff
    

    if model_name == 'Llama-3.2-1B-Instruct':
        tl_model_name = 'meta-llama/Llama-3.2-1B-Instruct'
    elif model_name == 'phi-1.5' or model_name == 'phi-1_5':
        tl_model_name = 'microsoft/phi-1_5'
    else:
        raise NotImplementedError

    # Find circuits on both the original model and the unlearned model
    # Ignore unlearn --> does not work
    for ori_or_unlearn in ['original', 'unlearn'][:1]:
        # if os.path.exists(f"saves/circuit/{ori_or_unlearn}/{task_name}.json"):
        #     continue

        print('Find circuit on:', ori_or_unlearn)
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
        model.cfg.use_split_qkv_input = True
        model.cfg.use_attn_result = True
        model.cfg.use_hook_mlp_in = True
        model.cfg.ungroup_grouped_query_attention = True
        model.eval()
        for n, p in model.named_parameters():
            p.requires_grad = False
            if 'embed' in n:
                print(n)
                p.requires_grad = True
            # print(p.requires_grad)

        # Find circuit
        g = Graph.from_model(model)
        scores = get_scores_ig_activations(model, g, data_loader, metric_fn, steps=5)
        g.to_json(f"saves/circuit/{ori_or_unlearn}/{task_name}.json")
        # g.to_json(f"saves/circuit/{ori_or_unlearn}/{task_name.replace('forget10', 'retain90')}.json")

        # for topn in [100, 200, 300, 400]:
        #     if os.path.exists(f"saves/circuit/{ori_or_unlearn}/{cfg.get('task_name')}_top{topn}_loss.csv"):
        #         continue

        #     else:
        #         g = Graph.from_json(f"saves/circuit/{ori_or_unlearn}/{cfg.get('task_name')}.json")
        #         g.apply_topn(topn, True)
        #         g.to_json(f"saves/circuit/{ori_or_unlearn}/{cfg.get('task_name')}_top{topn}.json")

        #         scores1 = evaluate_standard(model, data_loader, metric_fn)
        #         scores2 = evaluate_graph(model, g, data_loader, metric_fn, intervention='zero', ablate_non_circuit_edges=True)
        #         scores3 = evaluate_graph(model, g, data_loader, metric_fn, intervention='zero', ablate_non_circuit_edges=False)
                
        #         loss = pd.DataFrame({
        #             'idx': subset_samples,
        #             'post_unlearn_loss': scores1,
        #             'ablate_circuit_loss': scores2,
        #             'ablate_circuit_loss2': scores3,
        #         })
        #         loss.to_csv(f"saves/circuit/{ori_or_unlearn}/{cfg.get('task_name')}_top{topn}_loss.csv", index=None)


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
