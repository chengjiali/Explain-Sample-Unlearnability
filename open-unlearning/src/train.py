import sys
sys.path.append('/home/public/jcheng2/robust_unlearning')
import re
import wandb
import hydra
from omegaconf import DictConfig, OmegaConf
from data import get_data, get_collators
from model import get_model
from trainer import load_trainer
from evals import get_evaluators
from trainer.utils import seed_everything


def wandb_setup(cfg):
    data, model, split, unlearn_method, train_method, seed = cfg.get('task_name').strip().split('_')

    project = 'Mode Connectivity in Unlearning'
    group = 'train_endpoints'
    name = [data, split, model, unlearn_method, train_method, seed]
    tags = [f"Data={data}", f"Split={split}", f"Model={model}", 
            f"Unlearn_Method={unlearn_method}", f"Train_Method={train_method}", f"Seed={seed}"]

    name = '-'.join(name)
    run_id = name
    wandb.init(project=project, group=group, name=name, 
               config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True), 
               id=run_id, tags=tags, resume='allow')

@hydra.main(version_base=None, config_path="../configs", config_name="train.yaml")
def main(cfg: DictConfig):
    """Entry point of the code to train models
    Args:
        cfg (DictConfig): Config to train
    """
    # wandb_setup(cfg)
    seed_everything(cfg.trainer.args.seed)
    mode = cfg.get("mode", "train")
    model_cfg = cfg.model
    template_args = model_cfg.template_args
    assert model_cfg is not None, "Invalid model yaml passed in train config."
    model, tokenizer = get_model(model_cfg)

    # Load Dataset
    data_cfg = cfg.data
    data = get_data(
        data_cfg, mode=mode, tokenizer=tokenizer, template_args=template_args
    )

    # Load collator
    collator_cfg = cfg.collator
    collator = get_collators(collator_cfg, tokenizer=tokenizer)

    # Get Trainer
    trainer_cfg = cfg.trainer
    assert trainer_cfg is not None, ValueError("Please set trainer")

    # Get Evaluators
    evaluators = None
    eval_cfgs = cfg.get("eval", None)
    # if eval_cfgs:
    #     evaluators = get_evaluators(
    #         eval_cfgs=eval_cfgs,
    #         template_args=template_args,
    #         model=model,
    #         tokenizer=tokenizer,
    #     )

    cl_cfg = trainer_cfg.cl
    trainer, trainer_args = load_trainer(
        trainer_cfg=trainer_cfg,
        model=model,
        train_dataset=data.get("train", None),
        eval_dataset=data.get("eval", None),
        tokenizer=tokenizer,
        data_collator=collator,
        evaluators=evaluators,
        template_args=template_args,
        cl_cfg=cl_cfg
    )

    if trainer_args.do_train:
        trainer.train()
        trainer.save_state()
        trainer.save_model(trainer_args.output_dir)

    if trainer_args.do_eval:
        trainer.evaluate(metric_key_prefix="eval")


if __name__ == "__main__":
    main()
