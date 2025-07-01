# Mask2Former API Documentation

## Table of Contents

1. [Overview](#overview)
2. [Installation & Setup](#installation--setup)
3. [Core Components](#core-components)
4. [Configuration API](#configuration-api)
5. [Model Architecture](#model-architecture)
6. [Data Loading & Processing](#data-loading--processing)
7. [Training API](#training-api)
8. [Inference API](#inference-api)
9. [Evaluation API](#evaluation-api)
10. [Utilities](#utilities)
11. [Examples](#examples)

## Overview

Mask2Former is a unified architecture for panoptic, instance, and semantic segmentation. This documentation covers all public APIs, functions, and components available in the framework.

**Key Features:**
- Single architecture for multiple segmentation tasks
- Support for major datasets: ADE20K, Cityscapes, COCO, Mapillary Vistas
- Test-time augmentation support
- Video instance segmentation capabilities

## Installation & Setup

### Basic Setup

```python
from detectron2.config import get_cfg
from detectron2.projects.deeplab import add_deeplab_config
from mask2former import add_maskformer2_config

# Initialize configuration
cfg = get_cfg()
add_deeplab_config(cfg)
add_maskformer2_config(cfg)
```

### Required Dependencies

See `requirements.txt` for complete list:
- detectron2
- torch
- torchvision
- opencv-python
- scipy
- pycocotools

## Core Components

### 1. MaskFormer Model

The main model class for mask classification semantic segmentation architectures.

#### Class: `MaskFormer`

```python
from mask2former import MaskFormer

class MaskFormer(nn.Module):
    """
    Main class for mask classification semantic segmentation architectures.
    """
```

**Constructor Parameters:**
- `backbone` (Backbone): A backbone module following detectron2's interface
- `sem_seg_head` (nn.Module): Module for semantic segmentation prediction
- `criterion` (nn.Module): Loss computation module
- `num_queries` (int): Number of object queries
- `object_mask_threshold` (float): Threshold for query filtering in panoptic segmentation
- `overlap_threshold` (float): Overlap threshold for panoptic segmentation
- `metadata`: Dataset metadata for category information
- `size_divisibility` (int): Input size divisibility requirement
- `semantic_on` (bool): Enable semantic segmentation output
- `instance_on` (bool): Enable instance segmentation output
- `panoptic_on` (bool): Enable panoptic segmentation output

**Key Methods:**

##### `forward(batched_inputs)`

Performs forward pass for training or inference.

**Parameters:**
- `batched_inputs` (list): List of input dictionaries, each containing:
  - `"image"`: Tensor in (C, H, W) format
  - `"instances"`: Ground truth instances (training only)
  - `"height"`, `"width"`: Output resolution

**Returns:**
- Training: Dictionary of losses
- Inference: List of result dictionaries with keys:
  - `"sem_seg"`: Semantic segmentation logits (K×H×W)
  - `"panoptic_seg"`: Tuple of (panoptic_seg, segments_info)
  - `"instances"`: Instance segmentation results

**Example:**
```python
# Training
model = MaskFormer.from_config(cfg)
losses = model(batched_inputs)

# Inference
model.eval()
with torch.no_grad():
    results = model(batched_inputs)
```

##### `from_config(cfg)`

Class method to build model from configuration.

**Parameters:**
- `cfg`: Configuration object

**Returns:**
- Dictionary of model parameters for initialization

## Configuration API

### Function: `add_maskformer2_config(cfg)`

Adds Mask2Former-specific configuration options to the config object.

**Parameters:**
- `cfg`: Detectron2 configuration object

**Key Configuration Groups:**

#### Input Configuration
```python
cfg.INPUT.DATASET_MAPPER_NAME = "mask_former_semantic"  # Dataset mapper selection
cfg.INPUT.COLOR_AUG_SSD = False  # Color augmentation
cfg.INPUT.CROP.SINGLE_CATEGORY_MAX_AREA = 1.0  # Crop constraints
cfg.INPUT.SIZE_DIVISIBILITY = -1  # Padding requirements
cfg.INPUT.IMAGE_SIZE = 1024  # LSJ augmentation
cfg.INPUT.MIN_SCALE = 0.1  # Minimum scale
cfg.INPUT.MAX_SCALE = 2.0  # Maximum scale
```

#### Model Configuration
```python
# Loss weights
cfg.MODEL.MASK_FORMER.CLASS_WEIGHT = 1.0
cfg.MODEL.MASK_FORMER.DICE_WEIGHT = 1.0
cfg.MODEL.MASK_FORMER.MASK_WEIGHT = 20.0
cfg.MODEL.MASK_FORMER.NO_OBJECT_WEIGHT = 0.1

# Transformer configuration
cfg.MODEL.MASK_FORMER.NHEADS = 8
cfg.MODEL.MASK_FORMER.DROPOUT = 0.1
cfg.MODEL.MASK_FORMER.DIM_FEEDFORWARD = 2048
cfg.MODEL.MASK_FORMER.DEC_LAYERS = 6
cfg.MODEL.MASK_FORMER.HIDDEN_DIM = 256
cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES = 100

# Inference configuration
cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON = True
cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON = False
cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON = False
```

#### Swin Transformer Backbone
```python
cfg.MODEL.SWIN.PRETRAIN_IMG_SIZE = 224
cfg.MODEL.SWIN.PATCH_SIZE = 4
cfg.MODEL.SWIN.EMBED_DIM = 96
cfg.MODEL.SWIN.DEPTHS = [2, 2, 6, 2]
cfg.MODEL.SWIN.NUM_HEADS = [3, 6, 12, 24]
cfg.MODEL.SWIN.WINDOW_SIZE = 7
```

## Model Architecture

### 1. Criterion (Loss Computation)

#### Class: `SetCriterion`

Computes losses for DETR-style models using Hungarian matching.

```python
from mask2former.modeling.criterion import SetCriterion

criterion = SetCriterion(
    num_classes=num_classes,
    matcher=matcher,
    weight_dict=weight_dict,
    eos_coef=no_object_weight,
    losses=["labels", "masks"],
    num_points=cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS,
    oversample_ratio=cfg.MODEL.MASK_FORMER.OVERSAMPLE_RATIO,
    importance_sample_ratio=cfg.MODEL.MASK_FORMER.IMPORTANCE_SAMPLE_RATIO,
)
```

**Parameters:**
- `num_classes` (int): Number of object categories
- `matcher`: Hungarian matcher module
- `weight_dict` (dict): Loss weights
- `eos_coef` (float): No-object category weight
- `losses` (list): List of losses to apply ["labels", "masks"]
- `num_points` (int): Number of points for mask loss sampling
- `oversample_ratio` (float): Point sampling oversample ratio
- `importance_sample_ratio` (float): Importance sampling ratio

### 2. Hungarian Matcher

#### Class: `HungarianMatcher`

Computes optimal assignment between predictions and targets.

```python
from mask2former.modeling.matcher import HungarianMatcher

matcher = HungarianMatcher(
    cost_class=1.0,
    cost_mask=1.0,
    cost_dice=1.0,
    num_points=12544
)
```

**Parameters:**
- `cost_class` (float): Classification cost weight
- `cost_mask` (float): Mask focal loss weight  
- `cost_dice` (float): Dice loss weight
- `num_points` (int): Number of sampling points

## Data Loading & Processing

### Dataset Mappers

#### 1. Semantic Segmentation Mapper

```python
from mask2former import MaskFormerSemanticDatasetMapper

mapper = MaskFormerSemanticDatasetMapper(cfg, is_train=True)
```

#### 2. Instance Segmentation Mapper

```python
from mask2former import MaskFormerInstanceDatasetMapper

mapper = MaskFormerInstanceDatasetMapper(cfg, is_train=True)
```

#### 3. Panoptic Segmentation Mapper

```python
from mask2former import MaskFormerPanopticDatasetMapper

mapper = MaskFormerPanopticDatasetMapper(cfg, is_train=True)
```

#### 4. COCO Baseline Mappers

```python
from mask2former import COCOInstanceNewBaselineDatasetMapper, COCOPanopticNewBaselineDatasetMapper

# Instance segmentation with LSJ augmentation
instance_mapper = COCOInstanceNewBaselineDatasetMapper(cfg, is_train=True)

# Panoptic segmentation with LSJ augmentation  
panoptic_mapper = COCOPanopticNewBaselineDatasetMapper(cfg, is_train=True)
```

**Common Parameters:**
- `cfg`: Configuration object
- `is_train` (bool): Training or inference mode

## Training API

### Trainer Class

Extended trainer class for Mask2Former training.

```python
from train_net import Trainer

class Trainer(DefaultTrainer):
    """Extension of the Trainer class adapted to MaskFormer."""
```

#### Key Methods:

##### `build_evaluator(cfg, dataset_name, output_folder=None)`

Creates appropriate evaluator for the dataset.

**Parameters:**
- `cfg`: Configuration object
- `dataset_name` (str): Name of dataset
- `output_folder` (str, optional): Output directory

**Returns:**
- Evaluator instance or DatasetEvaluators

##### `build_train_loader(cfg)`

Builds training data loader with appropriate mapper.

**Parameters:**
- `cfg`: Configuration object

**Returns:**
- DataLoader instance

##### `build_optimizer(cfg, model)`

Builds optimizer with custom parameter grouping.

**Parameters:**
- `cfg`: Configuration object  
- `model`: Model instance

**Returns:**
- Optimizer instance

### Training Script Usage

```python
from train_net import main, setup
import argparse

# Setup arguments
args = argparse.Namespace(
    config_file="path/to/config.yaml",
    opts=[],
    eval_only=False,
    resume=False,
    num_gpus=1,
    num_machines=1,
    machine_rank=0,
    dist_url="auto"
)

# Run training
cfg = setup(args)
result = main(args)
```

## Inference API

### Basic Inference

```python
from detectron2.engine import DefaultPredictor

# Setup predictor
cfg.MODEL.WEIGHTS = "path/to/model.pkl"
cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON = True
cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON = True  
cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON = True

predictor = DefaultPredictor(cfg)

# Run inference
import cv2
image = cv2.imread("path/to/image.jpg")
outputs = predictor(image)
```

### Prediction Script (Cog Interface)

```python
from predict import Predictor

# Initialize predictor
predictor = Predictor()
predictor.setup()

# Run prediction
from pathlib import Path
result_path = predictor.predict(Path("input_image.jpg"))
```

### Test-Time Augmentation

```python
from mask2former import SemanticSegmentorWithTTA

# Wrap model with TTA
tta_model = SemanticSegmentorWithTTA(cfg, model, batch_size=1)

# Run inference with TTA
results = tta_model(batched_inputs)
```

**Parameters:**
- `cfg`: Configuration object
- `model`: Base model instance
- `tta_mapper` (callable, optional): Custom TTA mapper
- `batch_size` (int): Batch size for augmented images

## Evaluation API

### Instance Segmentation Evaluator

```python
from mask2former import InstanceSegEvaluator

evaluator = InstanceSegEvaluator(
    dataset_name="coco_2017_val",
    output_dir="./output"
)
```

**Parameters:**
- `dataset_name` (str): Dataset name
- `output_dir` (str): Output directory for results

### Built-in Evaluation

The trainer automatically selects appropriate evaluators based on dataset metadata:

- **Semantic Segmentation**: `SemSegEvaluator`
- **Instance Segmentation**: `COCOEvaluator`, `InstanceSegEvaluator`
- **Panoptic Segmentation**: `COCOPanopticEvaluator`
- **Cityscapes**: `CityscapesInstanceEvaluator`, `CityscapesSemSegEvaluator`

## Utilities

### Miscellaneous Utilities

```python
from mask2former.utils.misc import nested_tensor_from_tensor_list

# Convert list of tensors to nested tensor
tensor_list = [torch.randn(3, 100, 100), torch.randn(3, 120, 80)]
nested_tensor, mask = nested_tensor_from_tensor_list(tensor_list)
```

### Loss Functions

```python
from mask2former.modeling.criterion import dice_loss, sigmoid_ce_loss

# Dice loss for masks
loss_dice = dice_loss(predictions, targets, num_masks)

# Sigmoid cross-entropy loss  
loss_ce = sigmoid_ce_loss(predictions, targets, num_masks)
```

## Examples

### 1. Basic Training Setup

```python
from detectron2.config import get_cfg
from detectron2.projects.deeplab import add_deeplab_config
from mask2former import add_maskformer2_config
from train_net import Trainer

# Configuration
cfg = get_cfg()
add_deeplab_config(cfg)
add_maskformer2_config(cfg)
cfg.merge_from_file("configs/coco/panoptic-segmentation/swin/maskformer2_swin_large_IN21k_384_bs16_100ep.yaml")

# Training
trainer = Trainer(cfg)
trainer.resume_or_load(resume=False)
trainer.train()
```

### 2. Custom Dataset Integration

```python
from detectron2.data import DatasetCatalog, MetadataCatalog

# Register custom dataset
def get_custom_dicts():
    # Return list of dataset dictionaries
    pass

DatasetCatalog.register("custom_train", get_custom_dicts)
MetadataCatalog.get("custom_train").set(
    thing_classes=["class1", "class2"],
    evaluator_type="coco"
)

# Update config
cfg.DATASETS.TRAIN = ("custom_train",)
cfg.DATASETS.TEST = ("custom_val",)
```

### 3. Multi-task Inference

```python
# Configure for all tasks
cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON = True
cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON = True
cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON = True

predictor = DefaultPredictor(cfg)
outputs = predictor(image)

# Access different outputs
semantic_seg = outputs["sem_seg"]
instances = outputs["instances"] 
panoptic_seg, segments_info = outputs["panoptic_seg"]
```

### 4. Video Instance Segmentation

```python
# Use video-specific training script
python train_net_video.py \
    --config-file configs/youtubevis_2019/video_maskformer2_R50_bs16_8ep.yaml \
    --num-gpus 8
```

### 5. Custom Loss Function

```python
from mask2former.modeling.criterion import SetCriterion

# Custom loss weights
weight_dict = {
    "loss_ce": 2.0,
    "loss_mask": 5.0, 
    "loss_dice": 5.0
}

criterion = SetCriterion(
    num_classes=num_classes,
    matcher=matcher,
    weight_dict=weight_dict,
    eos_coef=0.1,
    losses=["labels", "masks"]
)
```

### 6. Visualization

```python
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
import cv2

# Load image and run inference
image = cv2.imread("image.jpg")
outputs = predictor(image)

# Visualize results
v = Visualizer(image[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)

# Panoptic segmentation
panoptic_result = v.draw_panoptic_seg(outputs["panoptic_seg"][0].to("cpu"), 
                                     outputs["panoptic_seg"][1]).get_image()

# Instance segmentation  
instance_result = v.draw_instance_predictions(outputs["instances"].to("cpu")).get_image()

# Semantic segmentation
semantic_result = v.draw_sem_seg(outputs["sem_seg"].argmax(0).to("cpu")).get_image()
```

## Advanced Usage

### Custom Backbone Integration

```python
# Register custom backbone
from detectron2.modeling import BACKBONE_REGISTRY

@BACKBONE_REGISTRY.register()
class CustomBackbone(Backbone):
    def __init__(self, cfg):
        super().__init__()
        # Implementation
        
    def forward(self, x):
        # Return dict of feature maps
        return {"res2": feat2, "res3": feat3, "res4": feat4, "res5": feat5}
```

### Custom Dataset Mapper

```python
from mask2former.data.dataset_mappers.mask_former_semantic_dataset_mapper import MaskFormerSemanticDatasetMapper

class CustomDatasetMapper(MaskFormerSemanticDatasetMapper):
    def __call__(self, dataset_dict):
        # Custom preprocessing
        dataset_dict = super().__call__(dataset_dict)
        # Additional transformations
        return dataset_dict
```

This comprehensive documentation covers all major APIs, functions, and components in the Mask2Former framework. For specific implementation details, refer to the source code and configuration files in the repository.