import os
import warnings

import hydra
import pandas as pd
import torch
import wandb
from omegaconf import OmegaConf
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies import DDPStrategy

from scripts.data_loader import train_val_test_loader
from scripts.exceptions import (
    EvaluateFreshInitializedModelException,
    UnknownModeException,
)
from scripts.model_config import model_selection
from scripts.utilis import model_path


def save_results_to_csv(results: list[dict], filename: str) -> None:
    """Save the evaluation results to a CSV file.

    Parameters
    ----------
    results : list[dict]
        List of dictionaries containing the evaluation results.
    filename : str, optional
        The filename or path where the results should be saved.
    """
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    base, ext = os.path.splitext(filename)
    if ext.lower() != ".csv":
        warnings.warn(
            f"Evaluation results file '{filename}',"
            f" extension '{ext}' is not '.csv'. Changing it to '.csv'."
        )
        filename = f"{base}.csv"

    counter = 1
    new_filename = filename
    while os.path.exists(new_filename):
        new_filename = f"{base}_{counter}{ext}"
        counter += 1

    df = pd.DataFrame(results)
    df.to_csv(new_filename, index=False)


def log_bar_charts_to_wandb(
    results: list[dict],
    mode: str,
) -> None:
    """
    Log bar charts to Wandb.

    Parameters
    ----------
    results : list[dict]
        List of dictionaries containing the evaluation results.
    mode : str
        The evaluation mode.
    """
    if mode == "all":
        metrics = {}
        for result in results:
            metric_name = result["metric"]
            key = f"{result['posterior']}_{result['step']}"
            if metric_name not in metrics:
                metrics[metric_name] = []
            metrics[metric_name].append([key, result["value"]])

        for metric_name, data in metrics.items():
            table = wandb.Table(data=data, columns=["Posterior_Step", metric_name])
            chart_key = f"bar_chart_{metric_name}"
            wandb.log({chart_key: wandb.plot.bar(table, "Posterior_Step", metric_name)})
    else:
        metrics = {}
        for result in results:
            metric_name = result["metric"]
            key = result.get("step", result.get("posterior"))
            if metric_name not in metrics:
                metrics[metric_name] = []
            metrics[metric_name].append([key, result["value"]])

        for metric_name, data in metrics.items():
            table = wandb.Table(data=data, columns=[mode.capitalize(), metric_name])
            chart_key = f"bar_chart_{metric_name}_{mode}"
            wandb.log(
                {chart_key: wandb.plot.bar(table, mode.capitalize(), metric_name)}
            )


def evaluate_model(cfg, model, trainer: Trainer, test_loader) -> None:
    """Evaluate the model based on the specified configuration.

    Parameters
    ----------
    cfg : Any
        Configuration object containing evaluation settings.
    model : Any
        The model to be evaluated.
    trainer : Trainer
        PyTorch Lightning Trainer object.
    test_loader : Any
        DataLoader for the test dataset.
    """
    results = []

    if cfg.evaluation.mode == "steps":
        for interference_step in cfg.evaluation.steps:
            model.diffusion.set_timesteps(interference_step)
            trainer.test(model, test_loader)
            metrics = trainer.callback_metrics
            for metric_name, metric_value in metrics.items():
                metric_name = metric_name.replace("test/", "")
                results.append(
                    {
                        "step": interference_step,
                        "metric": metric_name,
                        "value": metric_value.item(),
                    }
                )

    elif cfg.evaluation.mode == "posterior":
        for posterior in cfg.evaluation.posteriors:
            model.diffusion.set_posterior_type(posterior)
            trainer.test(model, test_loader)
            metrics = trainer.callback_metrics
            for metric_name, metric_value in metrics.items():
                metric_name = metric_name.replace("test/", "")
                results.append(
                    {
                        "posterior": posterior,
                        "metric": metric_name,
                        "value": metric_value.item(),
                    }
                )

    elif cfg.evaluation.mode == "all":
        for posterior in cfg.evaluation.posteriors:
            model.diffusion.set_posterior_type(posterior)
            for interference_step in cfg.evaluation.steps:
                model.diffusion.set_timesteps(interference_step)
                trainer.test(model, test_loader)
                metrics = trainer.callback_metrics
                for metric_name, metric_value in metrics.items():
                    metric_name = metric_name.replace("test/", "")
                    results.append(
                        {
                            "posterior": posterior,
                            "step": interference_step,
                            "metric": metric_name,
                            "value": metric_value.item(),
                        }
                    )
    else:
        raise UnknownModeException()

    log_bar_charts_to_wandb(results, cfg.evaluation.mode)

    if cfg.evaluation.save_results:
        save_results_to_csv(results, cfg.evaluation.results_file)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg) -> None:
    """Main function to evaluate the model based on the provided configuration.

    This function loads the model, sets up the logger, initializes the trainer,
    and evaluates the model using the specified configuration.

    Parameters
    ----------
    cfg : OmegaConf
        Configuration object containing all settings for model evaluation.

    Returns
    -------
    None

    Raises
    ------
    EvaluateFreshInitializedModelException
        If no pre-trained model is specified in the configuration.
    """

    if cfg.model.load_model is None:
        raise EvaluateFreshInitializedModelException()

    final_model_path = model_path(cfg)
    config_dict = OmegaConf.to_container(cfg, resolve=True)
    logger = WandbLogger(
        project=cfg.wandb.project,
        name=final_model_path.split("/")[-1],
        save_dir="logs",
        config=config_dict,
        entity=cfg.wandb.entity,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_gpus = torch.cuda.device_count()

    model = model_selection(cfg=cfg, device=device)
    _, _, test_loader = train_val_test_loader(cfg=cfg)

    model = model.to(device)

    trainer = Trainer(
        devices=num_gpus,
        logger=logger,
        strategy=DDPStrategy(find_unused_parameters=True),
    )

    evaluate_model(cfg, model, trainer, test_loader)


if __name__ == "__main__":
    main()
