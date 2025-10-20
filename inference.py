#!/usr/bin/env python3
"""
Simple inference script for Stem-JEPA model.
Loads a trained checkpoint and performs inference on audio files.
"""

import torch
import torchaudio
import numpy as np
from pathlib import Path
import argparse
from omegaconf import OmegaConf
import hydra
from lightning.pytorch import LightningModule
import matplotlib.pyplot as plt

# Define constant list of stems
stems = ["vocals", "bass", "drums", "other"]

def load_model_from_checkpoint(ckpt_path: str, config_path: str = None):
    """Load trained Stem-JEPA model from checkpoint."""
    ckpt_path = Path(ckpt_path)
    
    # Load experiment config
    if config_path is None:
        config_path = ckpt_path.parents[1] / "config.yaml"
    
    cfg = OmegaConf.load(config_path)
    
    # Instantiate model
    model: LightningModule = hydra.utils.instantiate(cfg.model)
    
    # Load checkpoint
    ckpt = torch.load(ckpt_path, map_location=torch.device("cpu"), weights_only=False)
    
    # Filter state dict to handle compiled modules
    state_dict = {}
    for key, value in ckpt["state_dict"].items():
        # Remove 'encoder._orig_mod.' prefix if present (from compiled models)
        new_key = key.replace("encoder._orig_mod.", "encoder.")
        new_key = new_key.replace("target_encoder._orig_mod.", "target_encoder.")
        new_key = new_key.replace("predictor._orig_mod.", "predictor.")
        state_dict[new_key] = value
    
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    
    return model, cfg

def preprocess_audio(audio_path: str, cfg, device="cpu", plot=False):
    """Load and preprocess audio file to spectrogram using the same transform as training."""
    # Load audio
    waveform, orig_sr = torchaudio.load(audio_path)
        
    # Resample if necessary
    target_sr = cfg.data.dataset.sample_rate
    if orig_sr != target_sr:
        resampler = torchaudio.transforms.Resample(orig_sr, target_sr)
        waveform = resampler(waveform)
    
    # Trim or pad to desired duration
    target_length = int(cfg.data.dataset.duration * target_sr)
    if waveform.shape[1] > target_length:
        # Trim to target length
        waveform = waveform[:, :target_length]
    elif waveform.shape[1] < target_length:
        # Pad with zeros
        padding = target_length - waveform.shape[1]
        waveform = torch.nn.functional.pad(waveform, (0, padding))
    waveform = waveform.to(device)

    # Create context and target masks
    idx = torch.tensor([0]).to(device)  # target index for predictor
    num_stems = waveform.shape[0]  # number of channels/stems
    mask = torch.ones(num_stems, dtype=torch.bool)  # context mask
    mask[idx] = False
    
    # Plot waveforms if requested
    if plot:
        plt.figure(figsize=(10, 4))
        plt.plot(np.arange(waveform.shape[1])/target_sr, waveform.t().cpu().numpy())
        plt.legend(stems)
        plt.title("Waveform")
        plt.xlabel("Time / s")
        plt.ylabel("Amplitude")
        plt.tight_layout()
        plt.show(block=False)

    # Divide waveform into context and target based on mask
    ctx_waveform = waveform[mask, ...].sum(dim=0)  # Sum of context stems
    tgt_waveform = waveform[idx, ...]  # Target stem

    # Apply the same transform as used during training
    transform = hydra.utils.instantiate(cfg.data.transform)
    transform = transform.to(device)

    # Convert to spectrogram
    ctx_spectrogram = transform(ctx_waveform.unsqueeze(0))  # [1, 1, F, T]
    tgt_spectrogram = transform(tgt_waveform.unsqueeze(0))  # [1, 1, F, T]

    return ctx_spectrogram, tgt_spectrogram, idx

def inference_on_audio(ckpt_path: str, audio_path: str, output_dir: str = None, plot: bool = False):
    """Run inference on a single audio file."""
    print(f"Loading model from: {ckpt_path}")
    model, cfg = load_model_from_checkpoint(ckpt_path)
    
    # Choose device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
   
    # Load audio and preprocess
    print(f"Processing audio: {audio_path}")
    ctx_spectrogram, tgt_spectrogram, idx = preprocess_audio(audio_path, cfg, device, plot=plot)
    print(f"Context spectrogram shape: {ctx_spectrogram.shape}")
    print(f"Target spectrogram shape: {tgt_spectrogram.shape}")
    print(f"Target index: {idx}")
    
    # Move everything to device
    model = model.to(device)
    ctx_spectrogram = ctx_spectrogram.to(device)
    tgt_spectrogram = tgt_spectrogram.to(device)
    
    # Check for nan values
    if torch.isnan(ctx_spectrogram).any() or torch.isnan(tgt_spectrogram).any():
        raise ValueError("Input spectrograms contain NaN values.")
    
    # Plot spectrograms if requested
    if plot:
        fig, axs = plt.subplots(2, 3, figsize=(12, 8))
        def plot_spectrogram(ax, spectrogram, title="Spectrogram"):
            spectrogram = spectrogram.squeeze().cpu()
            im = ax.imshow(spectrogram, aspect='auto', origin='lower')
            ax.set_title(title)
            ax.set_xlabel('Time')
            ax.set_ylabel('Frequency')
            plt.colorbar(im, ax=ax, format='%+2.0f dB')

        plot_spectrogram(axs[0, 0], ctx_spectrogram, title="Context Spectrogram")
        plot_spectrogram(axs[0, 1], tgt_spectrogram, title=f"Target Spectrogram ({stems[idx]})")
        
    # Extract features
    with torch.no_grad():
        latents = model.encoder(ctx_spectrogram)
        preds = model.predictor(latents, idx)
        targets = model.target_encoder(tgt_spectrogram)
    
    print(f"Extracted features:")
    print(f"  - Context features shape: {latents.shape}")
    print(f"  - Predicted features shape: {preds.shape}")
    print(f"  - Target features shape: {targets.shape}")

    # Plot features if requested
    if plot:
        def plot_features(ax, features, title="Features"):
            features = features.squeeze().cpu()
            im = ax.imshow(features, aspect='auto', origin='lower')
            ax.set_title(title)
            ax.set_xlabel('Time')
            ax.set_ylabel('Feature Dimension')
            plt.colorbar(im, ax=ax)

        plot_features(axs[1, 0], latents.sum(dim=1), title="Context Features")
        plot_features(axs[1, 1], preds.sum(dim=1), title="Predicted Features")
        plot_features(axs[1, 2], targets.sum(dim=1), title="Target Features")
        plot_features(axs[0, 2], (preds-targets).sum(dim=1), title="Prediction Error")

        plt.tight_layout()
        plt.show(block=True)
    
    # Save features if output directory provided
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        audio_name = Path(audio_path).stem
        
        # Save as numpy arrays
        np.save(output_dir / f"{audio_name}_target_features.npy", targets.cpu().numpy())
        np.save(output_dir / f"{audio_name}_predicted_features.npy", preds.cpu().numpy())

        print(f"Features saved to: {output_dir}")
    
    return 0

def main(ckpt_path=None, audio_path=None, output_dir=None):
    # If no arguments provided, use argparse for command line usage
    if ckpt_path is None or audio_path is None:
        parser = argparse.ArgumentParser(description="Stem-JEPA Inference")
        parser.add_argument("--ckpt_path", type=str, required=True,
                           help="Path to model checkpoint")
        parser.add_argument("--audio_path", type=str, required=True,
                           help="Path to audio file")
        parser.add_argument("--output_dir", type=str, default=None,
                           help="Directory to save extracted features")
        parser.add_argument("--config_path", type=str, default=None,
                           help="Path to config file (optional)")
        
        args = parser.parse_args()
        ckpt_path = args.ckpt_path
        audio_path = args.audio_path
        output_dir = args.output_dir
    
    # Run inference
    inference_on_audio(
        ckpt_path, 
        audio_path, 
        output_dir,
        plot=True
    )
    
    print("Inference completed successfully!")

if __name__ == "__main__":
    run = "a0d6f2db"
    ckpt_path = "/mnt/d/Dokumenty/GitHub/Stem-JEPA/logs/xps/" + run + "/checkpoints/last.ckpt"
    audio_path = "/mnt/d/Dokumenty/datasets/musdb18-16kHz-4ch-singlefile/A_Classic_Education_-_NightOwl.wav"
    output_dir = "/mnt/d/Dokumenty/GitHub/Stem-JEPA/logs/xps/" + run + "/features"
    main(ckpt_path, audio_path, output_dir)