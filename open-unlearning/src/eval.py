import re
import wandb
import hydra
from omegaconf import DictConfig, OmegaConf

import torch
from torch import Tensor
from torch.utils.data import DataLoader
from transformer_lens import HookedTransformer
from tqdm import tqdm
from einops import einsum

from eap.utils import tokenize_plus, make_hooks_and_matrices, compute_mean_activations
from eap.graph import Graph, AttentionNode

from trainer.utils import seed_everything
from model import get_model
from evals import get_evaluators


def wandb_setup(cfg):
    data, model, split, unlearn_method, train_method, seed = cfg.get('task_name').strip().split('_')

    project = 'Mode Connectivity in Unlearning'
    group = [data, split, model, unlearn_method, train_method, seed]
    name = [data, split, model, unlearn_method, train_method, seed]
    tags = [f"Experiment=train_endpoints", f"Data={data}", f"Split={split}", f"Model={model}", 
            f"Unlearn_Method={unlearn_method}", f"Train_Method={train_method}", f"Seed={seed}"]

    group = '-'.join(group)
    name = '-'.join(name)
    run_id = name
    wandb.init(project=project, group=group, name=name, 
               config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True), 
               id=run_id, tags=tags, resume='allow')

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


    ori_or_unlearn = 'original'
    # Parse MC configs
    task_name = cfg.get('task_name').strip()
    if 'phi-1_5' in task_name:
        task_name = task_name.replace('phi-1_5', 'phi-1.5')
    data_name, model_name, split, unlearn_method, hardness = task_name.split('_')

    if model_name == 'Llama-3.2-1B-Instruct':
        tl_model_name = 'meta-llama/Llama-3.2-1B-Instruct'
    elif model_name == 'phi-1.5' or model_name == 'phi-1_5':
        tl_model_name = 'microsoft/phi-1_5'
    else:
        raise NotImplementedError

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

    eval_cfgs = cfg.eval
    evaluators = get_evaluators(eval_cfgs)
    for evaluator_name, evaluator in evaluators.items():
        eval_args = {
            "template_args": template_args,
            "model": model,
            "tokenizer": tokenizer,
        }
        _ = evaluator.evaluate(**eval_args)


if __name__ == "__main__":
    main()
