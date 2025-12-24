"""
Training scheduler for multi-phase training
"""

import yaml
import subprocess
from pathlib import Path
from typing import Dict, List


class TrainingScheduler:
    """
    Scheduler for orchestrating multi-phase training
    """
    
    def __init__(self, config_path: str):
        self.config_path = config_path
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
    
    def run_phase(self, phase_name: str):
        """
        Run a specific training phase
        
        Args:
            phase_name: Name of the phase (phase_1_gan, phase_2_curriculum, phase_3_full)
        """
        phases = self.config['training']['phases']
        
        if phase_name not in phases:
            raise ValueError(f"Unknown phase: {phase_name}")
        
        phase_config = phases[phase_name]
        
        if not phase_config['enabled']:
            print(f"Phase {phase_name} is disabled, skipping...")
            return
        
        print(f"Starting phase: {phase_config['name']}")
        
        if phase_name == 'phase_1_gan':
            # Train GAN
            cmd = [
                'python', 'training/train_gan.py',
                '--config', self.config_path
            ]
        elif phase_name == 'phase_2_curriculum':
            # Train with curriculum learning
            cmd = [
                'python', 'training/train_yolo_cl.py',
                '--config', self.config_path
            ]
        elif phase_name == 'phase_3_full':
            # Train full model
            cmd = [
                'python', 'training/train_full_pipeline.py',
                '--config', self.config_path
            ]
        else:
            raise ValueError(f"Unknown phase: {phase_name}")
        
        # Run the training
        result = subprocess.run(cmd, check=True)
        print(f"Phase {phase_name} completed!")
    
    def run_all_phases(self):
        """Run all enabled phases in sequence"""
        phases = self.config['training']['phases']
        
        for phase_name in ['phase_1_gan', 'phase_2_curriculum', 'phase_3_full']:
            if phase_name in phases and phases[phase_name]['enabled']:
                self.run_phase(phase_name)
    
    def get_phase_status(self) -> Dict[str, bool]:
        """Get status of all phases"""
        phases = self.config['training']['phases']
        status = {}
        
        for phase_name, phase_config in phases.items():
            checkpoint_dir = Path(phase_config['checkpoint_dir'])
            status[phase_name] = {
                'enabled': phase_config['enabled'],
                'completed': checkpoint_dir.exists() and any(checkpoint_dir.glob('*.pth')),
            }
        
        return status


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Training scheduler")
    parser.add_argument("--config", type=str, default="config/train.yaml", help="Path to config file")
    parser.add_argument("--phase", type=str, help="Specific phase to run (optional)")
    parser.add_argument("--all", action="store_true", help="Run all phases")
    
    args = parser.parse_args()
    
    scheduler = TrainingScheduler(args.config)
    
    if args.phase:
        scheduler.run_phase(args.phase)
    elif args.all:
        scheduler.run_all_phases()
    else:
        print("Please specify --phase or --all")
        print("Available phases:", list(scheduler.config['training']['phases'].keys()))

