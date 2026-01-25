"""
Export RhythmMamba model to TorchScript for faster inference
TorchScript models don't require the original model code to run
"""

import os
import sys
import torch
from pathlib import Path

# Add to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from neural_methods.model.RhythmMamba import RhythmMamba


def find_latest_checkpoint():
    """Find the latest trained model checkpoint"""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    possible_paths = [
        os.path.join(base_dir, '..', '..', '..', 'experiment0', 'user', 'PreTrainedModels'),
        os.path.join(base_dir, 'checkpoints'),
        os.path.join(base_dir, 'results'),
    ]
    
    for search_dir in possible_paths:
        if not os.path.exists(search_dir):
            continue
        
        for root, dirs, files in os.walk(search_dir):
            for file in files:
                if 'UBFC_UBFC_UBFC_RhythmMamba_Epoch29.pth' in file:
                    return os.path.join(root, file)
                if 'RhythmMamba_Epoch' in file and file.endswith('.pth'):
                    checkpoint_files = [f for f in files if 'RhythmMamba_Epoch' in f and f.endswith('.pth')]
                    if checkpoint_files:
                        checkpoint_files.sort()
                        return os.path.join(root, checkpoint_files[-1])
    
    return None


def export_to_torchscript(checkpoint_path, output_path='rhythm_mamba.pt'):
    """Export model to TorchScript format"""
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Create model
    print("Creating RhythmMamba model...")
    model = RhythmMamba(
        img_size=128,
        patch_size=4,
        in_chans=3,
        embed_dim=144,
        depth=12,
        drop_rate=0.2
    )
    
    # Load checkpoint
    print(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Handle DataParallel format
    if isinstance(checkpoint, dict):
        state_dict = checkpoint
        if list(state_dict.keys())[0].startswith('module.'):
            print("Converting DataParallel checkpoint...")
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        
        model.load_state_dict(state_dict, strict=False)
    else:
        model.load_state_dict(checkpoint, strict=False)
    
    model = model.to(device)
    model.eval()
    
    print("✓ Model loaded")
    
    # Verify weights are loaded
    sample_param = next(model.parameters())
    print(f"✓ Weights loaded: param mean = {sample_param.abs().mean():.6f}")
    
    # Create dummy input for tracing
    print("\nTracing model with dummy input...")
    dummy_input = torch.randn(1, 160, 3, 128, 128).to(device)
    
    # Trace the model
    traced_model = torch.jit.trace(model, dummy_input)
    
    # Save to file
    print(f"Saving to {output_path}...")
    torch.jit.save(traced_model, output_path)
    
    # Verify the exported model
    print("\nVerifying exported model...")
    loaded_model = torch.jit.load(output_path, map_location=device)
    loaded_model.eval()
    
    with torch.no_grad():
        traced_output = loaded_model(dummy_input)
        original_output = model(dummy_input)
    
    print(f"Original output shape: {original_output.shape}")
    print(f"Traced output shape: {traced_output.shape}")
    print(f"Output difference: {(traced_output - original_output).abs().max():.8f}")
    
    print(f"\n✓ Model successfully exported to {output_path}")
    return output_path


if __name__ == '__main__':
    checkpoint_path = find_latest_checkpoint()
    
    if not checkpoint_path:
        print("ERROR: Could not find checkpoint!")
        sys.exit(1)
    
    print(f"Found checkpoint: {checkpoint_path}\n")
    
    output_path = os.path.join(os.path.dirname(__file__), 'rhythm_mamba.pt')
    export_to_torchscript(checkpoint_path, output_path)
