"""
Orchard demo application
"""

import yaml
import torch
import argparse
import cv2
import numpy as np
from pathlib import Path
import time

from models.yolo.yolo_attention import YOLOAttention
from utils.visualization import visualize_predictions


def load_config(config_path: str):
    """Load configuration"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def run_demo(config_path: str, checkpoint_path: str, camera_id: int = 0):
    """Run real-time detection demo"""
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
    
    # Open camera
    cap = cv2.VideoCapture(camera_id)
    
    print("Press 'q' to quit")
    
    fps_times = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        start_time = time.time()
        
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
        
        # Calculate FPS
        fps = 1.0 / (time.time() - start_time)
        fps_times.append(fps)
        if len(fps_times) > 30:
            fps_times.pop(0)
        avg_fps = np.mean(fps_times)
        
        # Add FPS text
        result_bgr = cv2.cvtColor(result_frame, cv2.COLOR_RGB2BGR)
        cv2.putText(
            result_bgr,
            f"FPS: {avg_fps:.1f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )
        
        cv2.imshow('Pomegranate Growth Stage Detection', result_bgr)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/train.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--camera", type=int, default=0)
    args = parser.parse_args()
    
    run_demo(args.config, args.checkpoint, args.camera)

