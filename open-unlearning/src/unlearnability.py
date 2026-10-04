import os
import wandb
import hydra
import torch
import torch.nn as nn
import pandas as pd
from tqdm import tqdm, trange
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from transformer_lens import HookedTransformer
from eap.graph import Graph
from eap.utils import make_hooks_and_matrices

from data.qa import QADataset
from data.collators import DataCollatorForSupervisedDataset

from trainer.utils import seed_everything
from model import get_model
import torch
import torch.nn.functional as F

def vectorize_circuit(graph):
    mask = graph.real_edge_mask
    in_graph = graph.in_graph.clone()[mask].bool()
    scores = graph.scores.clone()[mask].float()
    return scores * in_graph.float()

    
def cosine_sim(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """
    Cosine similarity between two matrices/tensors after flattening.
    Returns a scalar tensor.
    """
    a = a.reshape(-1).float()
    b = b.reshape(-1).float()

    # Handle degenerate (near-zero) vectors to avoid NaNs
    na = torch.norm(a, p=2)
    nb = torch.norm(b, p=2)
    if (na < eps) or (nb < eps):
        return torch.tensor(0.0, device=a.device)

    return torch.dot(a, b) / (na * nb + eps)


def anchored_unlearnability_score(
    Q: torch.Tensor,
    E: torch.Tensor,
    H: torch.Tensor,
    eps: float = 1e-12,
    clamp: bool = False,
) -> torch.Tensor:
    """
    Anchored score in [0,1] with guarantees:
      - score(E) = 0
      - score(H) = 1
    using the calibrated cosine construction:
        score = (1 - cos(Q,E)) / ((1 - cos(Q,E)) + (1 - cos(Q,H))).
    Returns a scalar tensor.
    """
    sE = cosine_sim(Q, E, eps=eps)
    sH = cosine_sim(Q, H, eps=eps)

    dE = (1.0 - sE).clamp_min(0.0)  # cosine-distance-ish
    dH = (1.0 - sH).clamp_min(0.0)

    denom = dE + dH
    score = torch.where(denom > eps, dE / (denom + eps), torch.tensor(0.5, device=Q.device))

    if clamp:
        score = score.clamp(0.0, 1.0)
    return score


def get_scores_ig_activations(model, graph, dataloader, metric, steps=5, quiet=False):

    # scores = torch.zeros((graph.n_forward, graph.n_backward), device='cpu', dtype=model.cfg.dtype)    
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

        nodelist_iter = nodeslist if quiet else tqdm(nodeslist, desc='Nodelist', leave=False)
        for node in nodelist_iter: # tqdm(nodeslist, desc='Nodelist', leave=False):
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
    data_name, model_name, split, unlearn_method = task_name.split('_')

    # Dataset
    if data_name == 'tofu':
        num_samples = 4000

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

    model = HookedTransformer.from_pretrained(
        tl_model_name,
        hf_model=None,
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


    # Find circuits on both the original model and the unlearned model
    # if os.path.exists(f"saves/circuit/{ori_or_unlearn}/{task_name}.json"):
    #     continue

    for i in trange(num_samples, desc='Evaluate Unlearnability'):
        if os.path.exists(f"saves/circuit/difficulty/{task_name}/{i}.json"):
            continue
        if data_name == 'tofu':
            hf_args = {'path': 'locuslab/TOFU', 'split': 'train'}
            dataset = QADataset(
                hf_args, template_args, tokenizer, max_length=512, predict_with_generate=False)
            collator = DataCollatorForSupervisedDataset(tokenizer, padding_side="right", index="index")
        
        dataset.data = dataset.data.select([i])
        data_loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collator)

        # Find circuit
        g = Graph.from_model(model)
        scores = get_scores_ig_activations(model, g, data_loader, metric_fn, steps=5, quiet=True)
        os.makedirs(f"saves/circuit/difficulty/{task_name}", exist_ok=True)
        g.to_json(f"saves/circuit/difficulty/{task_name}/{i}.json")

    # get_score(unlearn_method)


def get_score(m):

    ori = 'original'
    topn = 75
    g1 = Graph.from_json(f'saves/circuit/{ori}/tofu_Llama-3.2-1B-Instruct_forget10_{m}_easy.json')
    g2 = Graph.from_json(f'saves/circuit/{ori}/tofu_Llama-3.2-1B-Instruct_forget10_{m}_hard.json')

    g1.apply_greedy(topn)
    g2.apply_greedy(topn)

    easy = vectorize_circuit(g1)
    hard = vectorize_circuit(g2)

    res = []
    for i in trange(4000):
        if os.path.exists(f'saves/circuit/difficulty/tofu_Llama-3.2-1B-Instruct_forget10_{m}/{i}.pt'):
            continue

        g = Graph.from_json(f'saves/circuit/difficulty/tofu_Llama-3.2-1B-Instruct_forget10_{m}/{i}.json')
        g.apply_greedy(topn)

        q = vectorize_circuit(g)
        torch.save(q, f'saves/circuit/difficulty/tofu_Llama-3.2-1B-Instruct_forget10_{m}/{i}.pt')

        score = anchored_unlearnability_score(q, easy, hard)   
        res.append(score.item())

    with open(f'saves/circuit/difficulty/tofu_Llama-3.2-1B-Instruct_forget10_{m}/all.txt', 'w') as f:
        f.write('\n'.join([str(i) for i in res]))


if __name__ == "__main__":
    main()
