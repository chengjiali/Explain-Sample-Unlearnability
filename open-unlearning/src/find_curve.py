import hydra
from omegaconf import DictConfig
from data import get_data, get_collators
from model import get_model
from trainer import load_trainer
from evals import get_evaluator
from trainer.utils import seed_everything
import transformers
import torch
torch.autograd.set_detect_anomaly(True)



@hydra.main(version_base=None, config_path="../configs", config_name="train.yaml")
def main(cfg: DictConfig):
    """Entry point of the code to train models
    Args:
        cfg (DictConfig): Config to train
    """
    seed_everything(cfg.trainer.args.seed)
    mode = cfg.get("mode", "train")
    model_cfg = cfg.model
    template_args = model_cfg.template_args
    assert model_cfg is not None, "Invalid model yaml passed in train config."
    # model, tokenizer = get_model(model_cfg)

    init_start = f'{ckpt_dir}/unlearn/{unlearn_args.data_name}/{unlearn_args.backbone}/{unlearn_args.unlearn_method}/{unlearn_args.del_ratio}/42'
    init_end = f'{ckpt_dir}/unlearn/{unlearn_args.data_name}/{unlearn_args.backbone}/{unlearn_args.unlearn_method}/{unlearn_args.del_ratio}/87'

    if unlearn_args.data_name == 'tofu':
        midpoint_ckpt = f'locuslab/tofu_ft_{unlearn_config.backbone}'
    else:
        midpoint_ckpt = f'../MU-Bench/checkpoint/{unlearn_args.data_name}/{unlearn_args.backbone}'
    model = CurveModelLora(mubench.model_cls_map[unlearn_args.data_name], unlearn_args.mc_curve_type, init_start, init_end, midpoint_ckpt, unlearn_args.mc_num_bends)


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

    # Get Evaluator
    evaluator = None
    eval_cfgs = cfg.get("eval", None)
    # if eval_cfgs:
    #     assert len(eval_cfgs) <= 1, ValueError(
    #         "Only one evaluation supported while training"
    #     )
    #     eval_name, eval_cfg = next(iter(eval_cfgs.items()))
    #     evaluator = get_evaluator(
    #         eval_name,
    #         eval_cfg,
    #         template_args=template_args,
    #         model=model,
    #         tokenizer=tokenizer,
    #     )

    transformers.utils.logging.set_verbosity_info()
    trainer, trainer_args = load_trainer(
        trainer_cfg=trainer_cfg,
        model=model,
        train_dataset=data.get("train", None),
        eval_dataset=data.get("eval", None),
        tokenizer=tokenizer,
        data_collator=collator,
        evaluator=evaluator,
        template_args=template_args,
    )
    print('aaaaaaaaaa', trainer_args.output_dir)

    if trainer_args.do_train:
        trainer.train()
        trainer.save_state()
        trainer.save_model(trainer_args.output_dir)

    if trainer_args.do_eval:
        trainer.evaluate(metric_key_prefix="eval")


if __name__ == "__main__":
    main()
