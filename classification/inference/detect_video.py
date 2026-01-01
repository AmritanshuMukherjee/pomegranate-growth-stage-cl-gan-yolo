"""
Detect objects in video
"""

import yaml
import torch
import argparse
import cv2
import numpy as np
from pathlib import Path

from models.yolo.yolo_attention import YOLOAttention
from utils.visualization import visualize_predictions


def load_config(config_path: str):
    """Load configuration"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def detect_video(config_path: str, checkpoint_path: str, video_path: str, output_path: str):
    """Detect objects in video"""
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
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frame_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Preprocess
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        original_h, original_w = frame_rgb.shape[:2]
        
        image_resized = cv2.resize(frame_rgb, (dataset_config['image_size'], dataset_config['image_size']))
        image_tensor = torch.from_numpy(image_resized).permute(2, 0, 1).float() / 255.0
        image_tensor = image_tensor.unsqueeze(0).to(device)
        
        # Predict
        with torch.no_grad():
            predictions = model.predict(image_tensor, conf_threshold=0.25, iou_threshold=0.45)
        
        # Scale predictions
        if len(predictions) > 0 and len(predictions[0]) > 0:
            pred = predictions[0]
            if isinstance(pred, torch.Tensor):
                pred = pred.cpu().numpy()
            
            scale_x = original_w / dataset_config['image_size']
            scale_y = original_h / dataset_config['image_size']
            
            for p in pred:
                if len(p) >= 6:
                    p[0] *= scale_x
                    p[1] *= scale_y
                    p[2] *= scale_x
                    p[3] *= scale_y
        
        # Visualize
        result_frame = visualize_predictions(
            frame_rgb,
            predictions[0] if len(predictions) > 0 else torch.empty((0, 6)),
            class_names=dataset_config['class_names'],
        )
        
        # Convert to BGR and write
        result_bgr = cv2.cvtColor(result_frame, cv2.COLOR_RGB2BGR)
        out.write(result_bgr)
        
        if frame_count % 30 == 0:
            print(f"Processed {frame_count} frames")
    
    cap.release()
    out.release()
    print(f"Video saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/train.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--video", type=str, required=True)
    parser.add_argument("--output", type=str, default="result.mp4")
    args = parser.parse_args()
    
    detect_video(args.config, args.checkpoint, args.video, args.output)

