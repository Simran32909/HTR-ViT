import hydra
from omegaconf import DictConfig
import lightning as L
from lightning.pytorch.loggers import Logger
from typing import Any, Dict, List, Tuple
import torch

# This utility from the original repo helps with python path resolution
import rootutils
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
# Note: You may need to create an empty file named ".project-root" in the top-level
# directory of your new `htr-vit-iam` project for the line above to work.

from src.utils import (
    pylogger,
    rich_utils,
    instantiators,
    extras,
    task_wrapper,
    log_hyperparameters,
)

log = pylogger.RankedLogger(__name__, rank_zero_only=True)

# def nan_hook(module, input, output):
#     """A hook that checks for NaNs in the output of a module."""
#     if torch.isnan(output).any():
#         log.error(f"!!! NaN detected in the output of module: {module.__class__.__name__} !!!")
#         raise RuntimeError(f"NaN detected in module {module.__class__.__name__}")

@task_wrapper
def train(cfg: DictConfig) -> Tuple[Dict[str, float], Dict[str, Any]]:
    """Trains the model.
    This method is wrapped in @task_wrapper decorator, that controls the behavior during
    failure. Useful for multiruns, saving info about the crash, etc.
    """

    # set precision for tensor cores
    torch.set_float32_matmul_precision("medium")

    # Set all seeds for reproducibility
    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)

    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: L.LightningDataModule = hydra.utils.instantiate(cfg.data)

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: L.LightningModule = hydra.utils.instantiate(cfg.model)

    # Add the NaN hook to every submodule in the network
    # for name, module in model.net.named_modules():
    #     module.register_forward_hook(nan_hook)
    # log.info("NaN hook registered on all submodules.")

    # Instantiate callbacks
    log.info("Instantiating callbacks...")
    callbacks: List[L.Callback] = instantiators.instantiate_callbacks(cfg.get("callbacks"))

    # Instantiate loggers
    log.info("Instantiating loggers...")
    logger: List[Logger] = instantiators.instantiate_loggers(cfg.get("logger"))

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: L.Trainer = hydra.utils.instantiate(cfg.trainer, callbacks=callbacks, logger=logger)

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "callbacks": callbacks,
        "logger": logger,
        "trainer": trainer,
    }

    if logger:
        log.info("Logging hyperparameters!")
        log_hyperparameters(object_dict)

    if cfg.get("train"):
        log.info("Starting training!")
        trainer.fit(model=model, datamodule=datamodule, ckpt_path=cfg.get("ckpt_path"))

    train_metrics = trainer.callback_metrics

    if cfg.get("test"):
        log.info("Starting testing!")
        ckpt_path = trainer.checkpoint_callback.best_model_path
        if ckpt_path == "":
            log.warning("Best ckpt not found! Using current weights for testing...")
            ckpt_path = None
        trainer.test(model=model, datamodule=datamodule, ckpt_path=ckpt_path)
        log.info(f"Best ckpt path: {ckpt_path}")

    test_metrics = trainer.callback_metrics

    # merge train and test metrics
    metric_dict = {**train_metrics, **test_metrics}

    return metric_dict, object_dict


@hydra.main(version_base="1.3", config_path="../configs", config_name="config.yaml")
def main(cfg: DictConfig) -> None:
    # apply extra utilities like printing config and suppressing warnings
    extras(cfg)

    # train the model
    train(cfg)


if __name__ == "__main__":
    main() 