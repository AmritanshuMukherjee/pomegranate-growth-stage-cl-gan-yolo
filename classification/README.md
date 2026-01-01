<<<<<<< HEAD
# Attention-Guided Curriculum Learning with GAN-Enhanced YOLO for Multi-Source Pomegranate Growth Stage Classification

A comprehensive deep learning framework for detecting and classifying pomegranate growth stages using YOLO with attention mechanisms, curriculum learning, and GAN-based data augmentation.

## 📋 Overview

This project implements a multi-phase training pipeline for pomegranate growth stage detection:

1. **GAN Training**: Generate synthetic pomegranate images using Pix2Pix GAN
2. **Curriculum Learning**: Progressive training from easy to hard samples
3. **Attention Mechanisms**: CBAM attention for feature enhancement
4. **Full Pipeline**: Combined approach with all enhancements

## 🎯 Features

- **5 Growth Stages**: bud, flower, fruit_immature, fruit_mature, harvest
- **Multi-Source Dataset Support**: Handles multiple data sources
- **GAN Data Augmentation**: Pix2Pix for synthetic data generation
- **Curriculum Learning**: Progressive difficulty-based training
- **Attention Mechanisms**: CBAM, SE, CoordAttention, Self-Attention
- **Comprehensive Evaluation**: mAP, confusion matrix, cross-dataset evaluation
- **Real-time Inference**: Image, video, and camera demo

## 📁 Project Structure

```
pomegranate-growth-stage-cl-gan-yolo/
├── config/              # Configuration files
├── data/                # Dataset directories
├── models/              # Model implementations
│   ├── yolo/           # YOLO models
│   ├── attention/      # Attention modules
│   └── gan/            # GAN models
├── training/            # Training scripts
├── utils/               # Utility functions
├── evaluation/          # Evaluation scripts
├── inference/           # Inference scripts
└── experiments/        # Experiment outputs
```

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd pomegranate-growth-stage-cl-gan-yolo

# Install dependencies
pip install -r requirements.txt
```

### Dataset Preparation

1. Organize your dataset in YOLO format:
```
data/raw/source_1/
├── images/
│   ├── img1.jpg
│   └── ...
└── labels/
    ├── img1.txt
    └── ...
```

2. Update `config/dataset.yaml` with your dataset paths

### Training

#### Option 1: Full Pipeline (Recommended)
```bash
# Train all phases sequentially
python main.py --mode scheduler --config config/train.yaml

# Or train individual phases
python main.py --mode scheduler --config config/train.yaml --phase phase_1_gan
python main.py --mode scheduler --config config/train.yaml --phase phase_2_curriculum
python main.py --mode scheduler --config config/train.yaml --phase phase_3_full
```

#### Option 2: Individual Components
```bash
# Train GAN
python training/train_gan.py --config config/train.yaml

# Train baseline YOLO
python training/train_yolo_baseline.py --config config/train.yaml

# Train YOLO with attention
python training/train_yolo_attention.py --config config/train.yaml

# Train YOLO with curriculum learning
python training/train_yolo_cl.py --config config/train.yaml

# Train full pipeline
python training/train_full_pipeline.py --config config/train.yaml
```

### Evaluation

```bash
# Evaluate model
python main.py --mode eval --config config/train.yaml --checkpoint experiments/exp_005_full_model/checkpoints/best_model.pth

# Or use specific evaluation script
python evaluation/evaluate_full_model.py --config config/train.yaml --checkpoint <checkpoint_path>

# Generate confusion matrix
python evaluation/confusion_matrix.py --config config/train.yaml --checkpoint <checkpoint_path> --output confusion_matrix.png

# Cross-dataset evaluation
python evaluation/cross_dataset_eval.py --config config/train.yaml --checkpoint <checkpoint_path> --source source_2
```

### Inference

```bash
# Detect objects in image
python main.py --mode infer --config config/train.yaml --checkpoint <checkpoint_path> --image <image_path> --output result.jpg

# Or use specific inference script
python inference/detect_image.py --config config/train.yaml --checkpoint <checkpoint_path> --image <image_path> --output result.jpg

# Detect objects in video
python inference/detect_video.py --config config/train.yaml --checkpoint <checkpoint_path> --video <video_path> --output result.mp4

# Real-time camera demo
python inference/orchard_demo.py --config config/train.yaml --checkpoint <checkpoint_path> --camera 0
```

### Generate Synthetic Data

```bash
# Generate synthetic images using trained GAN
python training/generate_synthetic.py --config config/gan.yaml --checkpoint <gan_checkpoint_path>
```

## ⚙️ Configuration

All configuration is done through YAML files in the `config/` directory:

- `config/dataset.yaml`: Dataset configuration
- `config/yolo.yaml`: YOLO model configuration
- `config/gan.yaml`: GAN configuration
- `config/attention.yaml`: Attention mechanism configuration
- `config/curriculum.yaml`: Curriculum learning configuration
- `config/train.yaml`: Main training configuration

## 📊 Dataset

This project uses the **"A Dataset of Pomegranate Growth Stages (Zhao et al., 2023)"**:
- **5,857 images**
- **5 classes**: bud, flower, fruit_immature, fruit_mature, harvest
- **Public dataset** for pomegranate detection

## 🔬 Methodology

1. **GAN Phase**: Train Pix2Pix GAN to generate synthetic pomegranate images
2. **Curriculum Learning Phase**: Train YOLO progressively from easy to hard samples
3. **Full Model Phase**: Train YOLO with attention mechanisms and curriculum learning

## 📈 Results

The framework supports comprehensive evaluation:
- mAP@0.5 and mAP@0.5:0.95
- Per-class precision and recall
- Confusion matrices
- Cross-dataset generalization

## 🛠️ Model Export

```bash
# Export to ONNX
python inference/export_model.py --config config/train.yaml --checkpoint <checkpoint_path> --output model.onnx --format onnx

# Export to TorchScript
python inference/export_model.py --config config/train.yaml --checkpoint <checkpoint_path> --output model.pt --format torchscript
```

## 📝 Citation

If you use this code, please cite:

```bibtex
@article{zhao2023pomegranate,
  title={A Dataset of Pomegranate Growth Stages},
  author={Zhao et al.},
  year={2023},
  journal={Dataset paper (labeling)}
}
```

## 📄 License

[Add your license here]

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📧 Contact

[Add your contact information]

## 🙏 Acknowledgments

- YOLO architecture based on YOLOv8
- CBAM attention mechanism
- Pix2Pix GAN implementation
- Dataset: Zhao et al. (2023)

=======
# pomegranate-growth-stage-cl-gan-yolo
>>>>>>> 5ecb7d7a3445b87a8bf89284ce06db5515e6b80c
