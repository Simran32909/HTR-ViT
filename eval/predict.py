# This script will be used to run inference on a given dataset. 

import argparse
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import torch
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm
import wandb
from torchmetrics.text import CharErrorRate, WordErrorRate

from src.data.components.tokenizers import CharTokenizer
from src.models.crnn_ctc_module import CRNN_CTC_Module
from src.data.data_utils import collate_fn # We need this for resizing
from src.models.components.htr_vit import MaskedAutoencoderViT

def find_data_pairs(data_dir):
    """Scans a directory to find matching png and json files."""
    data_dir = Path(data_dir)
    image_files = list(data_dir.rglob("*.png"))
    pairs = []
    for img_path in image_files:
        json_path = img_path.with_suffix(".json")
        if json_path.exists():
            pairs.append({"image": str(img_path), "json": str(json_path)})
    return pairs

def main(args):
    # Initialize a new W&B run
    wandb.init(project=args.wandb_project, name=args.wandb_run_name, config=args)

    # --- Manually build the model and load weights ---

    # 1. The actual tokenizer
    tokenizer = CharTokenizer(model_name="char_tokenizer", vocab_file=args.vocab_path)

    # 2. Load the raw checkpoint to access the pickled model and hyperparameters
    print(f"Loading raw checkpoint from: {args.ckpt_path}")
    # Set weights_only=False as the checkpoint contains pickled class objects
    ckpt = torch.load(args.ckpt_path, map_location='cpu', weights_only=False)
    hparams = ckpt['hyper_parameters']

    # 3. Extract the already-instantiated network from the hyperparameters
    print("Extracting network from saved hyperparameters...")
    net = hparams['net']
    net.tokenizer = tokenizer # Ensure the network's tokenizer is our new instance

    # 4. Instantiate the parent Lightning Module (CRNN_CTC_Module)
    print("Instantiating LightningModule...")
    dummy_optimizer = torch.optim.Adam(net.parameters(), lr=1e-4)
    class DummyLogger:
        def log_hyperparams(self, params): pass
    dummy_logger = {"dummy": DummyLogger()}
    dummy_data_config = SimpleNamespace(datasets={})
    dummy_datamodule = SimpleNamespace(
        train_config=dummy_data_config, val_config=dummy_data_config, test_config=dummy_data_config
    )
    
    model = CRNN_CTC_Module(
        net=net, optimizer=dummy_optimizer, scheduler=None, compile=False,
        _logger=dummy_logger, datamodule=dummy_datamodule, tokenizer=tokenizer
    )

    # 5. Load the state dict 
    print("Loading weights into the compiled network …")
    state_dict = ckpt["state_dict"]
    # rename keys so that `net._orig_mod.xxx` → `net.xxx`
    state_dict = {k.replace("net._orig_mod.", "net."): v for k, v in state_dict.items()}
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"missing keys: {len(missing)}   unexpected keys: {len(unexpected)}")
    
    model.eval()
    
    # Set the device for inference (hardcoded to a specific GPU)
    device = torch.device("cuda:0")
        
    model.to(device)
    print(f"Model loaded on device: {device}")

    # The image size is now passed as an argument
    img_size = tuple(args.img_size)

    # Instantiate metric calculators
    cer_metric = CharErrorRate()
    wer_metric = WordErrorRate()

    # Find all image-json pairs in the OOD directory
    print(f"Scanning for data in: {args.ood_data_dir}")
    data_pairs = find_data_pairs(args.ood_data_dir)
    print(f"Found {len(data_pairs)} image-label pairs.")

    results = []
    # Create a W&B Table to log predictions
    prediction_table = wandb.Table(columns=["image", "ground_truth", "prediction", "cer", "wer"])

    for i, pair in enumerate(tqdm(data_pairs, desc="Running Inference")):
        # Load ground truth from JSON
        try:
            with open(pair["json"], 'r', encoding='utf-8') as f:
                # Use "original_text" as the key for the ground truth label
                ground_truth = json.load(f).get("original_text", "")
        except (json.JSONDecodeError, KeyError):
            ground_truth = "" # Handle cases with bad JSON or missing key

        # Load and preprocess the image
        image = Image.open(pair["image"]).convert("RGB")
        
        # Use collate_fn to preprocess exactly like training
        # collate_fn expects a list of (image_tensor, text) tuples
        transform = transforms.ToTensor()
        image_tensor = transform(image)
        
        # collate_fn will handle the resizing and padding
        images_batch, _, _ = collate_fn(
            [(image_tensor, "dummy")], 
            img_size, 
            lambda x: torch.tensor([0])  # dummy text transform
        )
        
        images_batch = images_batch.to(device)

        # Run the exact validation logic from the model
        with torch.no_grad():
            # This is the EXACT code from validation_step
            raw_preds = model.net(images_batch).squeeze(-1).clone()
            preds = raw_preds.clone().argmax(-1)
            
            # Process each prediction in the batch (we only have 1)
            for j in range(images_batch.shape[0]):
                _pred = torch.unique_consecutive(preds[j].detach()).cpu().numpy().tolist()
                _pred = [idx for idx in _pred if idx != model.net.vocab_size]  # Remove blank token
                predicted_text = tokenizer.detokenize(_pred)

        # Calculate metrics
        cer = cer_metric([predicted_text], [ground_truth]).item()
        wer = wer_metric([predicted_text], [ground_truth]).item()

        results.append({
            "image_path": pair["image"],
            "ground_truth": ground_truth,
            "prediction": predicted_text,
            "cer": cer,
            "wer": wer
        })

        # Add the first 200 results to the W&B Table
        if i < 200:
            prediction_table.add_data(
                wandb.Image(image),
                ground_truth,
                predicted_text,
                cer,
                wer
            )

    # Save results to a CSV file
    df = pd.DataFrame(results)
    df.to_csv(args.output_file, index=False, encoding='utf-8')
    print(f"Inference complete. Results saved to {args.output_file}")

    # Log the prediction table to W&B
    wandb.log({"OOD Predictions": prediction_table})

    # Log summary metrics
    mean_cer = df['cer'].mean()
    mean_wer = df['wer'].mean()
    wandb.log({
        "mean_ood_cer": mean_cer,
        "mean_ood_wer": mean_wer
    })

    # Log a histogram of CER values
    wandb.log({"cer_distribution": wandb.Histogram(df['cer'])})

    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference on an OOD dataset.")
    parser.add_argument(
        "--ckpt_path",
        type=str,
        required=True,
        help="Path to the model checkpoint (.ckpt) file."
    )
    parser.add_argument(
        "--ood_data_dir",
        type=str,
        required=True,
        help="Path to the root directory of the OOD dataset."
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="predictions.csv",
        help="Path to save the output CSV file."
    )
    parser.add_argument(
        "--vocab_path",
        type=str,
        required=True,
        help="Path to the vocabulary file used during training (e.g., sharada_vocab.txt)."
    )
    parser.add_argument(
        "--img_size",
        nargs=2,
        type=int,
        default=[64, 768],
        help="The image size (height width) used during training."
    )
    # Add W&B arguments
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="HTR-ViT-Inference",
        help="W&B project name."
    )
    parser.add_argument(
        "--wandb_run_name",
        type=str,
        default=None,
        help="W&B run name. A random one is generated if not provided."
    )
    args = parser.parse_args()
    main(args) 