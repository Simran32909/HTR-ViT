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

    if len(data_pairs) == 0:
        print("ERROR: No data pairs found! Check your data directory path.")
        wandb.finish()
        return

    results = []
    # Create a W&B Table to log predictions
    prediction_table = wandb.Table(columns=["image", "ground_truth", "ground_truth_clean", "prediction", "prediction_clean", "cer", "wer"])

    for i, pair in enumerate(tqdm(data_pairs, desc="Running Inference")):
        try:
            # Load ground truth from JSON
            with open(pair["json"], 'r', encoding='utf-8') as f:
                # Use "original_text" as the key for the ground truth label
                json_data = json.load(f)
                ground_truth = json_data.get("original_text", "")
                if not ground_truth:  # Try alternative keys if original_text is empty
                    ground_truth = json_data.get("text", json_data.get("label", ""))
                    
        except (json.JSONDecodeError, KeyError, FileNotFoundError) as e:
            print(f"Error reading JSON file {pair['json']}: {e}")
            ground_truth = "" # Handle cases with bad JSON or missing key

        # Load and preprocess the image
        try:
            image = Image.open(pair["image"]).convert("RGB")
        except Exception as e:
            print(f"Error loading image {pair['image']}: {e}")
            continue
        
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
                _pred = [idx for idx in _pred if idx != tokenizer.unk_id]  # Remove UNK token (ID=3)
                predicted_text = tokenizer.detokenize(_pred)

        # Remove UNK tokens from both predicted text and ground truth before computing CER
        # Additional cleanup in case any UNK tokens still remain after filtering at token level
        predicted_text_clean = predicted_text.replace("[UNK]", "")
        ground_truth_clean = ground_truth.replace("[UNK]", "")

        # Calculate metrics using cleaned texts
        cer = cer_metric([predicted_text_clean], [ground_truth_clean]).item()
        wer = wer_metric([predicted_text_clean], [ground_truth_clean]).item()

        results.append({
            "image_path": pair["image"],
            "ground_truth": ground_truth,
            "ground_truth_clean": ground_truth_clean,
            "prediction": predicted_text,
            "prediction_clean": predicted_text_clean,
            "cer": cer,
            "wer": wer
        })

        # Add the first 200 results to the W&B Table
        if i < 200:
            prediction_table.add_data(
                wandb.Image(image),
                ground_truth,
                ground_truth_clean,
                predicted_text,
                predicted_text_clean,
                cer,
                wer
            )

    # Check if we have any results
    if len(results) == 0:
        print("ERROR: No results were generated! Check that images can be processed and JSON files contain valid text.")
        wandb.finish()
        return

    # Save results to a CSV file
    df = pd.DataFrame(results)
    print(f"DataFrame shape: {df.shape}")
    print(f"DataFrame columns: {df.columns.tolist()}")
    print(f"First few rows:")
    print(df.head())
    
    df.to_csv(args.output_file, index=False, encoding='utf-8')
    print(f"Inference complete. Results saved to {args.output_file}")

    # Log the prediction table to W&B
    wandb.log({"OOD Predictions": prediction_table})

    # Log summary metrics - check if columns exist first
    if 'cer' in df.columns and 'wer' in df.columns:
        mean_cer = df['cer'].mean()
        mean_wer = df['wer'].mean()
        print(f"Mean CER: {mean_cer:.4f}")
        print(f"Mean WER: {mean_wer:.4f}")
        
        wandb.log({
            "mean_ood_cer": mean_cer,
            "mean_ood_wer": mean_wer
        })

        # Log a histogram of CER values
        wandb.log({"cer_distribution": wandb.Histogram(df['cer'])})
    else:
        print(f"ERROR: Expected columns 'cer' and 'wer' not found in DataFrame. Available columns: {df.columns.tolist()}")

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