# Mask2Former API 中文文档

## 目录

1. [概述](#概述)
2. [安装与设置](#安装与设置)
3. [核心组件](#核心组件)
4. [配置API](#配置api)
5. [模型架构](#模型架构)
6. [数据加载与处理](#数据加载与处理)
7. [训练API](#训练api)
8. [推理API](#推理api)
9. [评估API](#评估api)
10. [工具函数](#工具函数)
11. [使用示例](#使用示例)

## 概述

Mask2Former是一个统一的架构，可以同时进行全景分割、实例分割和语义分割。本文档涵盖了框架中所有可用的公共API、函数和组件。

**主要特性：**
- 单一架构支持多种分割任务
- 支持主要数据集：ADE20K、Cityscapes、COCO、Mapillary Vistas
- 支持测试时增强（TTA）
- 支持视频实例分割

## 安装与设置

### 基础设置

```python
from detectron2.config import get_cfg
from detectron2.projects.deeplab import add_deeplab_config
from mask2former import add_maskformer2_config

# 初始化配置
cfg = get_cfg()
add_deeplab_config(cfg)
add_maskformer2_config(cfg)
```

### 必需依赖

详见 `requirements.txt`：
- detectron2
- torch
- torchvision
- opencv-python
- scipy
- pycocotools

## 核心组件

### 1. MaskFormer 模型

用于掩码分类语义分割架构的主要模型类。

#### 类：`MaskFormer`

```python
from mask2former import MaskFormer

class MaskFormer(nn.Module):
    """
    掩码分类语义分割架构的主要类。
    """
```

**构造函数参数：**
- `backbone` (Backbone): 遵循detectron2接口的主干网络模块
- `sem_seg_head` (nn.Module): 语义分割预测模块
- `criterion` (nn.Module): 损失计算模块
- `num_queries` (int): 对象查询数量
- `object_mask_threshold` (float): 全景分割中查询过滤的阈值
- `overlap_threshold` (float): 全景分割的重叠阈值
- `metadata`: 类别信息的数据集元数据
- `size_divisibility` (int): 输入尺寸整除性要求
- `semantic_on` (bool): 启用语义分割输出
- `instance_on` (bool): 启用实例分割输出
- `panoptic_on` (bool): 启用全景分割输出

**主要方法：**

##### `forward(batched_inputs)`

执行训练或推理的前向传播。

**参数：**
- `batched_inputs` (list): 输入字典列表，每个包含：
  - `"image"`: (C, H, W) 格式的张量
  - `"instances"`: 真实标注实例（仅训练时）
  - `"height"`, `"width"`: 输出分辨率

**返回值：**
- 训练时：损失字典
- 推理时：结果字典列表，包含以下键：
  - `"sem_seg"`: 语义分割logits (K×H×W)
  - `"panoptic_seg"`: (panoptic_seg, segments_info) 元组
  - `"instances"`: 实例分割结果

**示例：**
```python
# 训练
model = MaskFormer.from_config(cfg)
losses = model(batched_inputs)

# 推理
model.eval()
with torch.no_grad():
    results = model(batched_inputs)
```

##### `from_config(cfg)`

从配置构建模型的类方法。

**参数：**
- `cfg`: 配置对象

**返回值：**
- 用于初始化的模型参数字典

## 配置API

### 函数：`add_maskformer2_config(cfg)`

向配置对象添加Mask2Former特定的配置选项。

**参数：**
- `cfg`: Detectron2配置对象

**主要配置组：**

#### 输入配置
```python
cfg.INPUT.DATASET_MAPPER_NAME = "mask_former_semantic"  # 数据集映射器选择
cfg.INPUT.COLOR_AUG_SSD = False  # 颜色增强
cfg.INPUT.CROP.SINGLE_CATEGORY_MAX_AREA = 1.0  # 裁剪约束
cfg.INPUT.SIZE_DIVISIBILITY = -1  # 填充要求
cfg.INPUT.IMAGE_SIZE = 1024  # LSJ增强
cfg.INPUT.MIN_SCALE = 0.1  # 最小缩放
cfg.INPUT.MAX_SCALE = 2.0  # 最大缩放
```

#### 模型配置
```python
# 损失权重
cfg.MODEL.MASK_FORMER.CLASS_WEIGHT = 1.0
cfg.MODEL.MASK_FORMER.DICE_WEIGHT = 1.0
cfg.MODEL.MASK_FORMER.MASK_WEIGHT = 20.0
cfg.MODEL.MASK_FORMER.NO_OBJECT_WEIGHT = 0.1

# Transformer配置
cfg.MODEL.MASK_FORMER.NHEADS = 8
cfg.MODEL.MASK_FORMER.DROPOUT = 0.1
cfg.MODEL.MASK_FORMER.DIM_FEEDFORWARD = 2048
cfg.MODEL.MASK_FORMER.DEC_LAYERS = 6
cfg.MODEL.MASK_FORMER.HIDDEN_DIM = 256
cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES = 100

# 推理配置
cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON = True
cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON = False
cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON = False
```

#### Swin Transformer 主干网络
```python
cfg.MODEL.SWIN.PRETRAIN_IMG_SIZE = 224
cfg.MODEL.SWIN.PATCH_SIZE = 4
cfg.MODEL.SWIN.EMBED_DIM = 96
cfg.MODEL.SWIN.DEPTHS = [2, 2, 6, 2]
cfg.MODEL.SWIN.NUM_HEADS = [3, 6, 12, 24]
cfg.MODEL.SWIN.WINDOW_SIZE = 7
```

## 模型架构

### 1. 损失计算（Criterion）

#### 类：`SetCriterion`

使用匈牙利匹配为DETR风格模型计算损失。

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

**参数：**
- `num_classes` (int): 对象类别数量
- `matcher`: 匈牙利匹配器模块
- `weight_dict` (dict): 损失权重
- `eos_coef` (float): 无对象类别权重
- `losses` (list): 要应用的损失列表 ["labels", "masks"]
- `num_points` (int): 掩码损失采样的点数
- `oversample_ratio` (float): 点采样过采样比率
- `importance_sample_ratio` (float): 重要性采样比率

### 2. 匈牙利匹配器

#### 类：`HungarianMatcher`

计算预测和目标之间的最优分配。

```python
from mask2former.modeling.matcher import HungarianMatcher

matcher = HungarianMatcher(
    cost_class=1.0,
    cost_mask=1.0,
    cost_dice=1.0,
    num_points=12544
)
```

**参数：**
- `cost_class` (float): 分类成本权重
- `cost_mask` (float): 掩码focal损失权重
- `cost_dice` (float): Dice损失权重
- `num_points` (int): 采样点数

## 数据加载与处理

### 数据集映射器

#### 1. 语义分割映射器

```python
from mask2former import MaskFormerSemanticDatasetMapper

mapper = MaskFormerSemanticDatasetMapper(cfg, is_train=True)
```

#### 2. 实例分割映射器

```python
from mask2former import MaskFormerInstanceDatasetMapper

mapper = MaskFormerInstanceDatasetMapper(cfg, is_train=True)
```

#### 3. 全景分割映射器

```python
from mask2former import MaskFormerPanopticDatasetMapper

mapper = MaskFormerPanopticDatasetMapper(cfg, is_train=True)
```

#### 4. COCO基线映射器

```python
from mask2former import COCOInstanceNewBaselineDatasetMapper, COCOPanopticNewBaselineDatasetMapper

# 带LSJ增强的实例分割
instance_mapper = COCOInstanceNewBaselineDatasetMapper(cfg, is_train=True)

# 带LSJ增强的全景分割
panoptic_mapper = COCOPanopticNewBaselineDatasetMapper(cfg, is_train=True)
```

**通用参数：**
- `cfg`: 配置对象
- `is_train` (bool): 训练或推理模式

## 训练API

### 训练器类

为Mask2Former训练扩展的训练器类。

```python
from train_net import Trainer

class Trainer(DefaultTrainer):
    """适配MaskFormer的训练器类扩展。"""
```

#### 主要方法：

##### `build_evaluator(cfg, dataset_name, output_folder=None)`

为数据集创建合适的评估器。

**参数：**
- `cfg`: 配置对象
- `dataset_name` (str): 数据集名称
- `output_folder` (str, 可选): 输出目录

**返回值：**
- 评估器实例或DatasetEvaluators

##### `build_train_loader(cfg)`

使用合适的映射器构建训练数据加载器。

**参数：**
- `cfg`: 配置对象

**返回值：**
- DataLoader实例

##### `build_optimizer(cfg, model)`

使用自定义参数分组构建优化器。

**参数：**
- `cfg`: 配置对象
- `model`: 模型实例

**返回值：**
- 优化器实例

### 训练脚本使用

```python
from train_net import main, setup
import argparse

# 设置参数
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

# 运行训练
cfg = setup(args)
result = main(args)
```

## 推理API

### 基础推理

```python
from detectron2.engine import DefaultPredictor

# 设置预测器
cfg.MODEL.WEIGHTS = "path/to/model.pkl"
cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON = True
cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON = True  
cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON = True

predictor = DefaultPredictor(cfg)

# 运行推理
import cv2
image = cv2.imread("path/to/image.jpg")
outputs = predictor(image)
```

### 预测脚本（Cog接口）

```python
from predict import Predictor

# 初始化预测器
predictor = Predictor()
predictor.setup()

# 运行预测
from pathlib import Path
result_path = predictor.predict(Path("input_image.jpg"))
```

### 测试时增强

```python
from mask2former import SemanticSegmentorWithTTA

# 用TTA包装模型
tta_model = SemanticSegmentorWithTTA(cfg, model, batch_size=1)

# 使用TTA运行推理
results = tta_model(batched_inputs)
```

**参数：**
- `cfg`: 配置对象
- `model`: 基础模型实例
- `tta_mapper` (callable, 可选): 自定义TTA映射器
- `batch_size` (int): 增强图像的批量大小

## 评估API

### 实例分割评估器

```python
from mask2former import InstanceSegEvaluator

evaluator = InstanceSegEvaluator(
    dataset_name="coco_2017_val",
    output_dir="./output"
)
```

**参数：**
- `dataset_name` (str): 数据集名称
- `output_dir` (str): 结果输出目录

### 内置评估

训练器会根据数据集元数据自动选择合适的评估器：

- **语义分割**: `SemSegEvaluator`
- **实例分割**: `COCOEvaluator`, `InstanceSegEvaluator`
- **全景分割**: `COCOPanopticEvaluator`
- **Cityscapes**: `CityscapesInstanceEvaluator`, `CityscapesSemSegEvaluator`

## 工具函数

### 杂项工具

```python
from mask2former.utils.misc import nested_tensor_from_tensor_list

# 将张量列表转换为嵌套张量
tensor_list = [torch.randn(3, 100, 100), torch.randn(3, 120, 80)]
nested_tensor, mask = nested_tensor_from_tensor_list(tensor_list)
```

### 损失函数

```python
from mask2former.modeling.criterion import dice_loss, sigmoid_ce_loss

# 掩码的Dice损失
loss_dice = dice_loss(predictions, targets, num_masks)

# Sigmoid交叉熵损失
loss_ce = sigmoid_ce_loss(predictions, targets, num_masks)
```

## 使用示例

### 1. 基础训练设置

```python
from detectron2.config import get_cfg
from detectron2.projects.deeplab import add_deeplab_config
from mask2former import add_maskformer2_config
from train_net import Trainer

# 配置
cfg = get_cfg()
add_deeplab_config(cfg)
add_maskformer2_config(cfg)
cfg.merge_from_file("configs/coco/panoptic-segmentation/swin/maskformer2_swin_large_IN21k_384_bs16_100ep.yaml")

# 训练
trainer = Trainer(cfg)
trainer.resume_or_load(resume=False)
trainer.train()
```

### 2. 自定义数据集集成

```python
from detectron2.data import DatasetCatalog, MetadataCatalog

# 注册自定义数据集
def get_custom_dicts():
    # 返回数据集字典列表
    pass

DatasetCatalog.register("custom_train", get_custom_dicts)
MetadataCatalog.get("custom_train").set(
    thing_classes=["类别1", "类别2"],
    evaluator_type="coco"
)

# 更新配置
cfg.DATASETS.TRAIN = ("custom_train",)
cfg.DATASETS.TEST = ("custom_val",)
```

### 3. 多任务推理

```python
# 配置所有任务
cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON = True
cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON = True
cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON = True

predictor = DefaultPredictor(cfg)
outputs = predictor(image)

# 访问不同输出
semantic_seg = outputs["sem_seg"]
instances = outputs["instances"] 
panoptic_seg, segments_info = outputs["panoptic_seg"]
```

### 4. 视频实例分割

```python
# 使用视频特定的训练脚本
python train_net_video.py \
    --config-file configs/youtubevis_2019/video_maskformer2_R50_bs16_8ep.yaml \
    --num-gpus 8
```

### 5. 自定义损失函数

```python
from mask2former.modeling.criterion import SetCriterion

# 自定义损失权重
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

### 6. 可视化

```python
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
import cv2

# 加载图像并运行推理
image = cv2.imread("image.jpg")
outputs = predictor(image)

# 可视化结果
v = Visualizer(image[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)

# 全景分割
panoptic_result = v.draw_panoptic_seg(outputs["panoptic_seg"][0].to("cpu"), 
                                     outputs["panoptic_seg"][1]).get_image()

# 实例分割
instance_result = v.draw_instance_predictions(outputs["instances"].to("cpu")).get_image()

# 语义分割
semantic_result = v.draw_sem_seg(outputs["sem_seg"].argmax(0).to("cpu")).get_image()
```

## 高级用法

### 自定义主干网络集成

```python
# 注册自定义主干网络
from detectron2.modeling import BACKBONE_REGISTRY

@BACKBONE_REGISTRY.register()
class CustomBackbone(Backbone):
    def __init__(self, cfg):
        super().__init__()
        # 实现
        
    def forward(self, x):
        # 返回特征图字典
        return {"res2": feat2, "res3": feat3, "res4": feat4, "res5": feat5}
```

### 自定义数据集映射器

```python
from mask2former.data.dataset_mappers.mask_former_semantic_dataset_mapper import MaskFormerSemanticDatasetMapper

class CustomDatasetMapper(MaskFormerSemanticDatasetMapper):
    def __call__(self, dataset_dict):
        # 自定义预处理
        dataset_dict = super().__call__(dataset_dict)
        # 额外变换
        return dataset_dict
```

## 总结

这份全面的中文文档涵盖了Mask2Former框架中的所有主要API、函数和组件。主要特点包括：

### **核心优势：**
- **统一架构**：单一模型同时支持全景、实例和语义分割
- **多数据集支持**：支持ADE20K、Cityscapes、COCO、Mapillary Vistas等主要数据集
- **灵活配置**：丰富的配置选项适应不同使用场景
- **测试时增强**：内置TTA支持提升性能
- **视频支持**：扩展的视频实例分割能力

### **使用场景：**
- 计算机视觉研究
- 图像分割应用开发
- 自动驾驶场景理解
- 医学图像分析
- 工业质检和缺陷检测

### **学习路径建议：**
1. 从基础推理示例开始
2. 理解配置系统和模型架构
3. 尝试自定义数据集训练
4. 探索高级功能如TTA和多任务学习
5. 根据需求定制组件

这个框架为图像分割任务提供了强大而灵活的解决方案，适合从研究到生产的各种应用场景。