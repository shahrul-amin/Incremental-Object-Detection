# Incremental Object Detection with ILOD

A comprehensive framework for Incremental object detection built on top of [Detectron2](https://github.com/facebookresearch/detectron2). This implementation provides an improved version of Faster-ILOD (Incremental Learning for Object Detection) with advanced knowledge distillation, attention mechanisms, and extensive analysis capabilities.

## Overview

This repository addresses the challenge of catastrophic forgetting in object detection when learning new classes incrementally. The system maintains performance on previously learned classes while adapting to new ones through a sophisticated knowledge distillation framework that operates at multiple levels of the detection pipeline.

## Methodology

### Core Architecture

The system is based on a teacher-student knowledge distillation framework where:

- **Teacher Model**: A frozen copy of the student model from the previous task, preserving knowledge of previously learned classes
- **Student Model**: The actively trained model learning new classes while maintaining performance on old ones
- **Multi-Level Distillation**: Knowledge transfer occurs at feature, RPN, and ROI head levels

### Knowledge Distillation Strategy

#### 1. Feature-Level Distillation
- Operates on backbone CNN features before region proposal generation
- Normalizes features to zero mean to reduce domain shift effects
- Applies selective masking to prevent negative transfer
- Enhanced with attention mechanisms for better feature alignment

#### 2. RPN-Level Distillation
- Distills region proposal network knowledge for object localization
- Includes both objectness classification and bounding box regression
- Uses confidence-based filtering to focus on high-quality proposals
- Maintains spatial understanding across learning tasks

#### 3. ROI-Head Distillation
- **Class-Selective Distillation**: Only applies distillation to previously learned classes
- Prevents interference with new class learning
- Supports both MSE and Huber loss functions
- Includes adaptive weighting to balance distillation with classification loss

### Advanced Features

#### Attention-Enhanced Feature Distillation
- Self-attention blocks applied to high-resolution features
- Learnable residual connections with gamma scaling
- Improves feature alignment between teacher and student models

#### Adaptive Loss Weighting
- Dynamic adjustment of distillation loss weights based on classification loss magnitude
- Prevents distillation from overwhelming new class learning
- Maintains balance between stability and plasticity

#### Teacher Model Correction
- Optional correction of teacher proposals based on current ground truth
- Reduces noise from teacher model predictions
- Configurable through `TEACHER_CORR` parameter

#### Progressive Huber Loss
- Gradual transition from MSE to Huber loss during training
- Exponential decay of delta parameter
- Provides robustness to outliers while maintaining sensitivity

## Workflow

### 1. Environment Setup
```bash
# Install Detectron2 and dependencies
pip install detectron2
pip install -r requirements.txt

# Register datasets
python -c "import datasets as ds; ds.register_common_datasets(voc=True, soda=True, coco=False)"
```

### 2. Dataset Preparation
- **VOC Dataset**: Configure class splits in `datasets/voc.py`
- **Custom Annotations**: Use utilities in `data/utils/` for dataset conversion
- **Incremental Splits**: Define task-specific class combinations

### 3. Training Pipeline

#### Task 1 (Base Learning)
```bash
python train.py configs/first_1.yaml --outdir output/task1/
```
- Trains on initial class set without distillation
- Establishes baseline model for subsequent tasks

#### Task N (Incremental Learning)
```bash
python train.py configs/last_N.yaml --outdir output/taskN/
```
- Loads weights from previous task
- Initializes teacher model from loaded weights
- Applies multi-level knowledge distillation
- Learns new classes while preserving old knowledge

### 4. Evaluation and Analysis
```bash
python train.py configs/taskN.yaml --test_only --dump_rpn rpn_analysis --dump_roi roi_analysis
```
- Comprehensive evaluation on all learned classes
- Generates detailed analysis files for RPN and ROI performance
- Supports backward and forward transfer analysis

## Configuration System

### Base Configuration Structure
```yaml
DATASETS:
  TRAIN: ("dataset_name",)
  TEST: ("test_dataset",)

MODEL:
  META_ARCHITECTURE: 'ILODRCNN'
  ROI_HEADS:
    NAME: 'ILODRoiHead'
    NUM_CLASSES: N
  PROPOSAL_GENERATOR:
    NAME: 'ILODRPN'

ILOD:
  DISTILLATION: True/False
  LOAD_TEACHER: True/False
  FEATURE_LAM: 1.0    # Feature distillation weight
  RPN_LAM: 1.0        # RPN distillation weight
  ROI_LAM: 1.0        # ROI distillation weight
  HUBER: True         # Use Huber loss
  TEACHER_CORR: True  # Enable teacher correction
```

### Key Parameters

- **DISTILLATION**: Enable/disable ILOD framework
- **LOAD_TEACHER**: Load teacher weights from checkpoint (for resuming training)
- **FEATURE_LAM**: Controls feature-level distillation strength
- **RPN_LAM**: Controls RPN-level distillation strength
- **ROI_LAM**: Controls ROI-head distillation strength
- **HUBER**: Switch between MSE and Huber loss for distillation
- **TEACHER_CORR**: Enable teacher proposal correction

## Data Augmentation

### Augmentation Pipeline
The framework includes a comprehensive data augmentation system designed for object detection:

- **Geometric Transformations**: Horizontal flip, rotation, scaling
- **Photometric Augmentations**: Brightness, contrast, saturation adjustments
- **Noise Injection**: Gaussian noise, blur effects
- **Bbox-Aware Processing**: All augmentations preserve bounding box annotations

### Usage
```python
# Enable data augmentation in config
DATA_AUGMENTATION: True

# Configure augmentation factor
AUGMENTATION_FACTOR: 5  # Generate 5 augmented versions per image
```

## Logging

### Performance Monitoring
- **Incremental Learning Metrics**: Backward transfer, forward transfer, forgetting measures
- **Component-wise Analysis**: Separate evaluation of RPN and ROI head performance
- **Class-specific Metrics**: Per-class AP tracking across learning tasks

### Debug and Visualization
- **RPN Proposal Logging**: Save raw proposals for analysis
- **ROI Prediction Logging**: Store soft predictions for distillation analysis
- **Feature Visualization**: Tools for analyzing learned representations

## Experimental Configurations

### Supported Scenarios
1. **Class-Incremental**: Learning new object classes sequentially
2. **Task-Incremental**: Learning new tasks with different class distributions
3. **Domain-Incremental**: Adapting to new visual domains

### Benchmark Datasets
- **PASCAL VOC**: Standard splits for Incremental learning evaluation
- **MS COCO**: Large-scale object detection with class incremental setups
- **SODA10M**: Autonomous driving dataset for domain adaptation

### Evaluation Protocols
- **Standard AP Metrics**: mAP across all learned classes
- **Stability-Plasticity Trade-off**: Balance between old and new class performance
- **Computational Efficiency**: Memory usage and inference time analysis

## Advanced Usage

### Custom Dataset Integration
1. Implement dataset loader in `datasets/`
2. Register dataset with Detectron2
3. Configure class mappings and splits
4. Add evaluation metrics

### Model Customization
- Extend `ILODRCNN` for custom architectures
- Modify distillation losses in `calc_*_distil_loss` functions
- Add new attention mechanisms or regularization techniques

### Hyperparameter Tuning
- Adjust distillation weights based on task difficulty
- Modify warmup iterations for stable training
- Tune adaptive weighting parameters

## Output Structure

```
output/
├── task1/
│   ├── model_final.pth
│   ├── metrics.json
│   └── tensorboard_logs/
├── task2/
│   ├── model_final.pth
│   ├── rpn_analysis/
│   ├── roi_analysis/
│   └── evaluation_results/
└── analysis/
    ├── Incremental_learning_metrics.json
    ├── class_performance_tracking.csv
    └── visualization_plots/
```

## Performance Considerations

### Memory Optimization
- Teacher model weights are frozen to reduce memory usage
- Gradient computation disabled for teacher network
- Efficient feature storage for analysis

### Training Efficiency
- Progressive learning rates for different tasks
- Warmup periods for distillation losses
- Adaptive batch sizing based on memory constraints

### Scalability
- Support for large-scale datasets through efficient data loading
- Modular architecture for easy extension
- GPU memory management for high-resolution images

## Research Applications

This framework supports research in:
- Incremental learning for computer vision
- Knowledge distillation in object detection
- Catastrophic forgetting mitigation
- Multi-task learning in vision systems
- Domain adaptation for object detection

## Citation and References

This implementation extends and improves upon:
- Faster-ILOD: [Pattern Recognition Letters, 2020]
- Knowledge Distillation for Object Detection
- Incremental Learning methodologies
- Detectron2 framework architecture