"""
Export model to ONNX or TorchScript
"""

import yaml
import torch
import argparse
import onnx
from pathlib import Path

from models.yolo.yolo_attention import YOLOAttention


def load_config(config_path: str):
    """Load configuration"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def export_onnx(config_path: str, checkpoint_path: str, output_path: str):
    """Export model to ONNX"""
    config = load_config(config_path)
    dataset_config = config['dataset']
    yolo_config = config['yolo']
    attention_config = config['attention']
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Create model
    model = YOLOAttention(
        num_classes=dataset_config['num_classes'],
        input_size=dataset_config['image_size'],
        model_size=yolo_config['model']['size'],
        attention_type=attention_config['type'],
        attention_config={
            'reduction_ratio': attention_config['cbam']['reduction_ratio'],
            'kernel_size': attention_config['cbam']['kernel_size'],
            'use_channel': attention_config['cbam']['use_channel'],
            'use_spatial': attention_config['cbam']['use_spatial'],
        },
    )
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    # Dummy input
    dummy_input = torch.randn(1, 3, dataset_config['image_size'], dataset_config['image_size']).to(device)
    
    # Export
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}},
        opset_version=11,
    )
    
    print(f"Model exported to {output_path}")


def export_torchscript(config_path: str, checkpoint_path: str, output_path: str):
    """Export model to TorchScript"""
    config = load_config(config_path)
    dataset_config = config['dataset']
    yolo_config = config['yolo']
    attention_config = config['attention']
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Create model
    model = YOLOAttention(
        num_classes=dataset_config['num_classes'],
        input_size=dataset_config['image_size'],
        model_size=yolo_config['model']['size'],
        attention_type=attention_config['type'],
        attention_config={
            'reduction_ratio': attention_config['cbam']['reduction_ratio'],
            'kernel_size': attention_config['cbam']['kernel_size'],
            'use_channel': attention_config['cbam']['use_channel'],
            'use_spatial': attention_config['cbam']['use_spatial'],
        },
    )
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    # Dummy input
    dummy_input = torch.randn(1, 3, dataset_config['image_size'], dataset_config['image_size']).to(device)
    
    # Export
    traced_model = torch.jit.trace(model, dummy_input)
    traced_model.save(output_path)
    
    print(f"Model exported to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/train.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--format", type=str, choices=['onnx', 'torchscript'], default='onnx')
    args = parser.parse_args()
    
    if args.format == 'onnx':
        export_onnx(args.config, args.checkpoint, args.output)
    else:
        export_torchscript(args.config, args.checkpoint, args.output)

