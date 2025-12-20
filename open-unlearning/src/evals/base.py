import os
import json
import logging
from evals.metrics import get_metrics
import copy
import wandb
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

logger = logging.getLogger("evaluator")


class Evaluator:
    def __init__(self, name, eval_cfg, **kwargs):
        self.name = name
        self.eval_cfg = eval_cfg
        self.metrics_cfg = self.eval_cfg.metrics
        self.metrics = self.load_metrics(self.metrics_cfg)
        logger.info(
            f"Evaluations stored in the experiment directory: {self.eval_cfg.output_dir}"
        )

    def get_logs_file_path(self, output_dir, suffix="EVAL"):
        """Returns the path to json file to store results"""
        logs_filename = os.path.join(output_dir, f"{self.name}_{suffix}.json")
        return logs_filename

    def load_logs_from_file(self, file):
        """Returns the cache of existing results"""
        logs = {}
        if os.path.exists(file):
            logger.info(f"Loading existing evaluations from {file}")
            with open(file, "r") as f:
                logs = json.load(f)
        return logs

    def save_logs(self, logs, file):
        """Save the logs in a json file"""
        logs = dict(sorted(logs.items()))
        os.makedirs(os.path.dirname(file), exist_ok=True)
        try:
            with open(file, "w") as f:
                json.dump(logs, f, indent=4)
        except Exception as e:
            raise RuntimeError(f"Failed to save {file}: {e}")

    def prepare_model(self, model):
        """Prepare model for evaluation"""
        model.eval()
        return model

    def load_metrics(self, metrics_cfg):
        """Load metrics for evaluation"""
        metrics = get_metrics(metrics_cfg)
        return metrics

    def summarize(self, logs):
        """Summarize the metrics results"""
        metric_summary = {}
        for metric_name, metric_results in logs.items():
            # if metric_name not in self.metrics:
            #     continue
            agg_value = metric_results.get("agg_value", None)
            if agg_value is not None:
                metric_summary[metric_name] = agg_value
        return metric_summary

    def evaluate(self, model, output_dir=None, overwrite=None, t=None, **kwargs):
        # set flag to overwrite metrics
        overwrite = self.eval_cfg.overwrite if overwrite is None else overwrite

        # Prepare model for evaluation
        model = self.prepare_model(model)

        # Set output_dir and file to store results
        output_dir = output_dir if output_dir else self.eval_cfg.output_dir
        logs_file_path = self.get_logs_file_path(output_dir)
        summary_file_path = self.get_logs_file_path(output_dir, suffix="SUMMARY")

        if t is not None:
            logs_file_path = logs_file_path.replace('.json', f'_{t}.json')
            summary_file_path = summary_file_path.replace('.json', f'_{t}.json')

        # Load existing results from file if any.
        logs = self.load_logs_from_file(logs_file_path) if not overwrite else {}

        logger.info(f"***** Running {self.name} evaluation suite *****")
        logger.info(f"Fine-grained evaluations will be saved to: {logs_file_path}")
        logger.info(
            f"Aggregated evaluations will be summarised in: {summary_file_path}"
        )
        for metric_name, metric_fn in self.metrics.items():
            if not overwrite and metric_name in logs and logs[metric_name]:
                logger.info(f"Skipping {metric_name}, already evaluated.")
                if "agg_value" in logs[metric_name]:
                    logger.info(
                        f"Result for metric {metric_name}:\t{logs[metric_name]['agg_value']}"
                    )
                self.save_logs(self.summarize(logs), summary_file_path)
                continue
            _ = logs.pop(metric_name, None)  # overwriting existing evals if present
            kwargs = {
                "tokenizer": kwargs.get("tokenizer", None),
                "template_args": kwargs.get("template_args", None),
            }
            metrics_args = self.eval_cfg.metrics[metric_name]
            _
            result = metric_fn(
                model,
                metric_name=metric_name,
                cache=logs,
                **kwargs,
                **metrics_args,
            )
            if "agg_value" in result:
                logger.info(f"Result for metric {metric_name}:\t{result['agg_value']}")
            self.save_logs(logs, logs_file_path)
            self.save_logs(self.summarize(logs), summary_file_path)

        return logs # self.summarize(logs)

    def evaluate_mode_connectivity_curve(self, model, output_dir=None, start=0, end=1, num_points=16, **kwargs):
        """
        Evaluate the model along the mode connectivity curve between two solutions.

        Args:
            model: A model object with `.interpolate_weights(t)` and `.final_model.load_state_dict(...)`.
            output_dir (str): Path to store curve evaluation results.
            start (float): Start value of t (typically 0.0).
            end (float): End value of t (typically 1.0).
            num_points (int): Number of samples between start and end.
            kwargs: Passed to the `evaluate` method.
        """

        wandb.define_metric("t")
        if hasattr(self, 'metrics'):
            for metric in self.metrics:
                wandb.define_metric(metric, step_metric="t")
        else:
            for task in self.tasks:
                task_name = self.get_task_name(task)
                wandb.define_metric(task_name, step_metric="t")

        output_dir = output_dir or self.eval_cfg.output_dir
        ts = np.linspace(start, end, num_points)
        all_metrics = []

        logger.info("***** Evaluating along mode connectivity curve *****")
        for idx, t in enumerate(tqdm(ts, desc="Mode Connectivity Evaluation")):

            # Interpolate model weights at t
            model = model.cpu()
            interpolated_weights = model.interpolate_weights(t)
            model.final_model.load_state_dict(interpolated_weights, strict=False)

            # logger.info(
            #     f'Eval with interpolated weights, t = {t}, '
            #     f'weight sum = {self.sum_model_weights(model.final_model)}, '
            #     f'and weight = {model.curve(t).detach().tolist()}'
            # )

            tmp_model = copy.deepcopy(model.final_model)
            tmp_model = tmp_model.cuda()

            # Evaluate this specific model; logs and summaries are handled by evaluate()
            summary = self.evaluate(tmp_model, output_dir=output_dir, overwrite=True, t=idx, **kwargs)
            tmp_model = tmp_model.cpu()
            del tmp_model, interpolated_weights
            torch.cuda.empty_cache()

            summary['t'] = float(t)
            all_metrics.append(summary)

            # 1. Log structured per-t metrics
            for k, v in summary.items():
                if k != "t":
                    wandb.log({f"test/{idx}/{k}": v})

            # 2. Log flat structure for shared plot: t vs metrics
            wandb.log({"t": float(t), **{k: v for k, v in summary.items() if k != "t"}})

        # Save results to CSV
        curve_path = os.path.join(output_dir, "eval_curve.csv")
        df = pd.DataFrame(all_metrics)
        df.to_csv(curve_path, index=False)
        logger.info(f"Saved mode connectivity evaluation results to {curve_path}")

        return df
