import torch
from torchmetrics.text import CharErrorRate, WordErrorRate
from typing import List, Dict, Any

# A placeholder for a generic logger object, to avoid wandb dependency here.
class BaseLogger:
    def log_metrics(self, metrics: Dict[str, Any]):
        pass
    def log(self, metrics: Dict[str, Any]):
        pass

class MetricLogger:
    """A custom logger class to handle aggregation and logging of metrics."""
    def __init__(self, logger: Any, tokenizer: Any, train_datasets: List[str], val_datasets: List[str], test_datasets: List[str]):
        self.logger = logger if logger else BaseLogger()
        self.tokenizer = tokenizer
        self.train_datasets = train_datasets
        self.val_datasets = val_datasets
        self.test_datasets = test_datasets
        self.epoch = 0

        # Metrics
        self.train_loss = []
        self.val_cers = {ds: CharErrorRate() for ds in self.val_datasets}
        self.val_wers = {ds: WordErrorRate() for ds in self.val_datasets}
        self.test_cers = {ds: CharErrorRate() for ds in self.test_datasets}
        self.test_wers = {ds: WordErrorRate() for ds in self.test_datasets}

    def update_epoch(self, epoch: int):
        self.epoch = epoch

    def log_learning_rate(self, lr: float, epoch: int):
        self.logger.log_metrics({"learning_rate": lr, "epoch": epoch})

    def log_images(self, images: torch.Tensor, caption: str):
        # This check is a bit specific, but avoids errors if not using wandb
        if hasattr(self.logger, 'log') and callable(self.logger.log):
            try:
                import wandb
                # Log the first 4 images to avoid clutter
                log_images = images[:4].permute(0, 2, 3, 1).cpu().numpy()
                self.logger.log({f"images/{caption}": [wandb.Image(img) for img in log_images]})
            except (ImportError, AttributeError):
                # wandb might not be installed or logger might not be a wandb logger
                pass

    def log_train_step(self, loss: torch.Tensor, *args, **kwargs):
        self.train_loss.append(loss.item())

    def log_train_metrics(self):
        avg_loss = sum(self.train_loss) / len(self.train_loss) if self.train_loss else 0
        self.logger.log_metrics({"train/loss_epoch": avg_loss, "epoch": self.epoch})
        self.train_loss.clear()

    def _update_metrics(self, cer_dict, wer_dict, pred, label, dataset):
        if dataset in cer_dict:
            cer_dict[dataset].update(pred, label)
        if dataset in wer_dict:
            wer_dict[dataset].update(pred, label)

    def log_val_step_cer(self, pred: str, label: str, dataset: str):
        self._update_metrics(self.val_cers, self.val_wers, pred, label, dataset)

    def log_val_step_wer(self, pred: str, label: str, dataset: str):
        self._update_metrics(self.val_cers, self.val_wers, pred, label, dataset)

    def log_val_metrics(self):
        val_cers_logged = {}
        all_cers = []
        for ds, cer_metric in self.val_cers.items():
            cer_val = cer_metric.compute()
            all_cers.append(cer_val)
            val_cers_logged[ds] = cer_val
            self.logger.log_metrics({f"val/cer_{ds}": cer_val, "epoch": self.epoch})
            cer_metric.reset()

        for ds, wer_metric in self.val_wers.items():
            wer_val = wer_metric.compute()
            self.logger.log_metrics({f"val/wer_{ds}": wer_val, "epoch": self.epoch})
            wer_metric.reset()
        
        avg_cer = sum(all_cers) / len(all_cers) if all_cers else 0
        
        # Fulfilling the expected return signature from crnn_ctc_module.py
        # This is a simplified logic.
        in_domain_cer = val_cers_logged.get(self.train_datasets[0], 0.0) if self.train_datasets else 0.0
        out_of_domain_cer = 0.0 # Placeholder
        heldout_domain_cers = {} # Placeholder
        return avg_cer, in_domain_cer, out_of_domain_cer, heldout_domain_cers, val_cers_logged

    def log_test_step_cer(self, pred: str, label: str, dataset: str):
         self._update_metrics(self.test_cers, self.test_wers, pred, label, dataset)
    
    def log_test_step_wer(self, pred: str, label: str, dataset: str):
         self._update_metrics(self.test_cers, self.test_wers, pred, label, dataset)

    def log_test_metrics(self):
        test_cers_logged = {}
        test_wers_logged = {}
        for ds, cer_metric in self.test_cers.items():
            cer_val = cer_metric.compute()
            test_cers_logged[ds] = cer_val
            self.logger.log_metrics({f"test/cer_{ds}": cer_val, "epoch": self.epoch})
            cer_metric.reset()

        for ds, wer_metric in self.test_wers.items():
            wer_val = wer_metric.compute()
            test_wers_logged[ds] = wer_val
            self.logger.log_metrics({f"test/wer_{ds}": wer_val, "epoch": self.epoch})
            wer_metric.reset()
        
        return test_cers_logged, test_wers_logged
    
    def log_val_step_confidence(self, *args, **kwargs):
        pass

    def log_val_step_calibration(self, *args, **kwargs):
        pass

    def log_test_step_confidence(self, *args, **kwargs):
        pass

    def log_test_step_calibration(self, *args, **kwargs):
        pass 