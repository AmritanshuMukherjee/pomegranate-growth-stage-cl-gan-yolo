"""
Main entry point for Pomegranate Growth Stage Detection
"""

import argparse
import yaml
from pathlib import Path
from training.scheduler import TrainingScheduler


def main():
    parser = argparse.ArgumentParser(
        description="Attention-Guided Curriculum Learning with GAN-Enhanced YOLO for Multi-Source Pomegranate Growth Stage Classification"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default="config/train.yaml",
        help="Path to training configuration file"
    )
    
    parser.add_argument(
        "--mode",
        type=str,
        choices=['train', 'eval', 'infer', 'gan', 'scheduler'],
        default='train',
        help="Mode: train, eval, infer, gan, or scheduler"
    )
    
    parser.add_argument(
        "--phase",
        type=str,
        help="Training phase (phase_1_gan, phase_2_curriculum, phase_3_full)"
    )
    
    parser.add_argument(
        "--checkpoint",
        type=str,
        help="Path to model checkpoint"
    )
    
    parser.add_argument(
        "--image",
        type=str,
        help="Path to input image for inference"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        help="Path to output file"
    )
    
    args = parser.parse_args()
    
    if args.mode == 'scheduler':
        # Run training scheduler
        scheduler = TrainingScheduler(args.config)
        if args.phase:
            scheduler.run_phase(args.phase)
        else:
            scheduler.run_all_phases()
    
    elif args.mode == 'train':
        # Training mode
        from training.train_full_pipeline import train_full_pipeline
        train_full_pipeline(args.config)
    
    elif args.mode == 'gan':
        # GAN training
        from training.train_gan import train_gan
        train_gan(args.config)
    
    elif args.mode == 'eval':
        # Evaluation mode
        if not args.checkpoint:
            print("Error: --checkpoint required for evaluation")
            return
        
        from evaluation.evaluate_full_model import evaluate_full_model
        evaluate_full_model(args.config, args.checkpoint)
    
    elif args.mode == 'infer':
        # Inference mode
        if not args.checkpoint or not args.image:
            print("Error: --checkpoint and --image required for inference")
            return
        
        from inference.detect_image import detect_image
        output_path = args.output or "result.jpg"
        detect_image(args.config, args.checkpoint, args.image, output_path)
    
    else:
        print(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    main()

