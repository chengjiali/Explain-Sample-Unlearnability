import os
import hydra
from omegaconf import DictConfig, open_dict

from trainer.utils import seed_everything
from model import get_model
from evals.compute_sample_difficulty import EvaluatorComputeSampleDifficulty


# import debugpy
# debugpy.listen(('localhost', 5678))
# print('waiting for debugger attach...')
# debugpy.wait_for_client()
# debugpy.breakpoint()


@hydra.main(version_base=None, config_path="../configs", config_name="eval.yaml")
def main(cfg: DictConfig):
    """Entry point of the code to evaluate models
    Args:
        cfg (DictConfig): Config to train
    """
    seed_everything(cfg.seed)
    task_name = cfg.get('task_name')
    model_name = task_name.split('_')[1]
    trainer = task_name.split('_')[-1]

    model_cfg = cfg.model
    model_args = model_cfg.model_args
    # with open_dict(model_args):
    #     ckpt_path = model_args["pretrained_model_name_or_path"]
    # 'Llama-3.2-1B-Instruct' if 'tofu' in ckpt_path else ckpt_path.split('_')[1]
    # if len(ckpt_path.split('_')) > 3:
    #     trainer = ckpt_path.split('_')[3]
    # else:
    #     trainer = 'Original'

    template_args = model_cfg.template_args
    assert model_cfg is not None, "Invalid model yaml passed in train config."
    model, tokenizer = get_model(model_cfg)


    if 'tofu' in task_name:
        dataset_name = 'tofu'
        data_split = cfg.get('forget_split')

    elif 'muse' in task_name:
        dataset_name = 'muse'
        data_split = cfg.get('data_split')

    elif 'wmdp' in task_name:
        dataset_name = 'wmdp'
        data_split = cfg.get('data_split')

    # eval_cfgs = cfg.eval
    # for evaluator_name, eval_cfg in eval_cfgs.items():
    output_dir = os.path.join('saves/sample_difficulty', dataset_name, model_name, data_split, trainer)
    eval_args = {
        "template_args": template_args,
        "model": model,
        "tokenizer": tokenizer,
        "output_dir": output_dir
    }
    evaluator = EvaluatorComputeSampleDifficulty(dataset_name, data_split, cfg, **eval_args)
    _ = evaluator.compute_sample_difficulty(**eval_args)


if __name__ == "__main__":
    main()
