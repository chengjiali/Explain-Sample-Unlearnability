import hydra
from omegaconf import DictConfig

from trainer.utils import seed_everything
from model import get_model
from evals import get_evaluators


@hydra.main(version_base=None, config_path="../configs", config_name="eval.yaml")
def main(cfg: DictConfig):
    """Entry point of the code to evaluate models
    Args:
        cfg (DictConfig): Config to train
    """
    seed_everything(cfg.seed)
    model_cfg = cfg.model
    template_args = model_cfg.template_args
    assert model_cfg is not None, "Invalid model yaml passed in train config."
    model, tokenizer = get_model(model_cfg, is_eval=True)

    eval_cfgs = cfg.eval
    evaluators = get_evaluators(eval_cfgs)

    sys_prompt = '''You are a model that knows everything about {}.'''
    
    if cfg.data_split in ['Books', 'books']:
        sys_prompt = sys_prompt.format('Harry Potter, the book series')
        template_args['user_start_tag'] = sys_prompt + ' ' + template_args['user_start_tag']

    elif cfg.data_split in ['News', 'news']:
        sys_prompt = sys_prompt.format('CNN News')
        template_args['user_start_tag'] = sys_prompt + ' ' + template_args['user_start_tag']

    else:
        template_args['system_prompt'] = sys_prompt.format('author profiles')


    for evaluator_name, evaluator in evaluators.items():
        eval_args = {
            "template_args": template_args,
            "model": model,
            "tokenizer": tokenizer,
        }
        _ = evaluator.evaluate(**eval_args)


if __name__ == "__main__":
    main()
