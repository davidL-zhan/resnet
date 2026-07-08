# Food101 图像分类项目

本项目围绕 Hugging Face `ethz/food101` 数据集做 101 类食物图片分类。仓库里同时保留了手写 ResNet、AlexNet、ViT 学习实现，以及当前正在训练入口中使用的 Hugging Face ViT 分类模型。

当前代码状态需要先看清楚：

- `train.py`：当前主训练入口，读取 Food101 并训练 `ViT.py` 里的 `ViTForImageClassification`。
- `ViT.py`：基于 `transformers.ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")` 的 Food101 分类模型。
- `VIT_learn.py`：手写 ViT 学习版，包含 Patch Embedding、手写多头自注意力、Transformer Block 和 `vit_tiny/vit_small/vit_base` 工厂函数；当前没有接入 `train.py`。
- `resnet.py`：手写 ResNet-18 / ResNet-34 / ResNet-50 实现，目前训练入口里保留了相关函数和预训练权重加载代码，但模型创建处已经被注释掉。
- `AlexNet.py`：独立 AlexNet 结构演示文件，当前没有接入 Food101 训练入口。
- `predict.py`：按 checkpoint 里的 `model_name` 重建 ResNet 做单图预测，当前只适配 ResNet checkpoint，不适配 `train.py` 现在保存的 ViT checkpoint。
- `download.py`：下载/检查 Food101，并保留 mean/std 统计函数；当前 `main()` 只打印数据集信息，mean/std 统计调用被注释掉。

## 1. 项目结构

```text
.
├── train.py              # 当前 Food101 训练入口，实际训练 ViTForImageClassification
├── ViT.py                # Hugging Face ViT backbone + 自定义分类头
├── VIT_learn.py          # 手写 ViT 学习实现，未接入训练入口
├── resnet.py             # 手写 ResNet-18/34/50
├── AlexNet.py            # AlexNet 结构演示
├── predict.py            # ResNet checkpoint 单图预测脚本
├── download.py           # Food101 数据集下载/检查脚本
├── requirements.txt      # 依赖列表
├── pretrained/           # 本地 ResNet ImageNet 预训练权重
├── checkpoints/          # 训练输出目录
└── pic/                  # 示例 Food101 图片
```

## 2. 环境和依赖

本项目使用 `resnet` conda 环境。激活环境后安装依赖：

```powershell
conda activate resnet
pip install -r requirements.txt
```

`requirements.txt` 当前包含 PyTorch CUDA 12.8 wheel 源，以及：

- `torch`
- `torchvision`
- `datasets`
- `transformers`
- `torchinfo`
- `tqdm`
- `pillow` 相关能力由图像读取链路使用

其中 `fastapi`、`uvicorn`、`streamlit`、`openai`、`requests` 也在依赖文件里，但当前 Food101 训练/预测主链路没有直接使用这些包。

## 3. 数据集

项目使用 Hugging Face 数据集：

```python
load_dataset("ethz/food101")
```

Food101 是 101 类食物图像分类数据集，Hugging Face 版本常用 split 为：

- `train`：75,750 张图片
- `validation`：25,250 张图片
- `image`：PIL 图片对象
- `label`：类别 id，范围是 `0` 到 `100`

训练脚本会把图片统一转成 RGB，并整理成：

```text
pixel_values: [B, 3, 224, 224]
labels:       [B]
logits:       [B, 101]
```

## 4. 检查数据集

运行：

```powershell
python download.py
```

当前 `download.py` 会：

1. 读取 `ethz/food101`。
2. 打印 dataset 结构。
3. 打印类别数量、前 10 个类别名、第一张图片尺寸和第一条标签。
4. 打开第一张训练图片。

文件里还保留了 `compute_mean_std()` 和 `print_stats_result()`，可以统计 Food101 train split 的 RGB mean/std；但当前 `main()` 里这两行是注释状态：

```python
# result = compute_mean_std(dataset["train"], args)
# print_stats_result(result, args.output)
```

如果要重新统计 mean/std，需要先在 `download.py` 里恢复这两行调用。

## 5. 当前训练入口

运行 `train.py` 会加载 Food101，并创建：

```python
model = ViTForImageClassification(num_classes=len(class_names))
```

也就是说，虽然 `train.py` 的命令行参数还保留了 `--model resnet18/resnet34/resnet50`，但当前实际训练模型不是 ResNet，而是 `ViT.py` 中的 Hugging Face ViT。

推荐当前按 ViT 训练时先不要传 `--pretrained` 或 `--pretrained-path`。原因是：

- `ViT.py` 已经通过 `ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")` 加载 ViT 预训练 backbone。
- `train.py` 里的 `--pretrained` 逻辑是给本项目手写 ResNet 加载本地 ImageNet `.pth` 权重的旧逻辑。
- 当前模型创建已经改成 ViT 后，ResNet `.pth` 权重不再是匹配的 ViT 权重。

基础训练命令：

```powershell
python train.py `
  --epochs 30 `
  --batch-size 32 `
  --lr 0.0003 `
  --amp `
  --output-dir checkpoints/food101_vit
```

如果 Windows 上 DataLoader 多进程出问题，先把 worker 数改成 0：

```powershell
python train.py `
  --epochs 30 `
  --batch-size 32 `
  --num-workers 0 `
  --lr 0.0003 `
  --amp `
  --output-dir checkpoints/food101_vit
```

训练输出：

- `last.pt`：最后一个 epoch 的 checkpoint。
- `best.pt`：验证集 top-1 最好的 checkpoint。
- `metrics.jsonl`：每个 epoch 的训练 loss、训练 top-1、验证 loss、验证 top-1 和历史 best top-1。

继续训练：

```powershell
python train.py `
  --resume checkpoints/food101_vit/last.pt `
  --epochs 60 `
  --output-dir checkpoints/food101_vit
```

## 6. 当前 ViT 模型

`ViT.py` 中的模型结构很短：

```text
输入图片 [B, 3, 224, 224]
  -> transformers.ViTModel
  -> 取 last_hidden_state[:, 0, :] 作为 cls token 特征
  -> Linear(hidden_size, 101)
  -> logits [B, 101]
```

核心类：

```python
class ViTForImageClassification(torch.nn.Module):
    def __init__(self, num_classes: int = 1000):
        self.vit = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
        self.classifier = torch.nn.Linear(self.hidden_size, num_classes)
```

这个文件依赖 `transformers`，首次运行时需要能访问 Hugging Face 或已经有本地缓存。

## 7. 手写 ViT 学习版

`VIT_learn.py` 是从零手写 ViT 的学习实现，主要模块包括：

- `PatchEmbedding`
- `MLP`
- `MultiHeadSelfAttention`
- `TransformerEncoderBlock`
- `VisionTransformer`
- `vit_tiny_patch16_224`
- `vit_small_patch16_224`
- `vit_base_patch16_224`

它的主数据流是：

```text
[B, 3, 224, 224]
  -> patch embedding: [B, 196, embed_dim]
  -> 拼接 cls token: [B, 197, embed_dim]
  -> 加 position embedding
  -> Transformer blocks
  -> 取 cls token
  -> classifier
  -> [B, num_classes]
```

当前 `train.py` 没有使用 `VIT_learn.py`。如果后续要训练手写 ViT，需要把 `train.py` 里的模型创建从 `ViTForImageClassification` 改成 `VIT_learn.py` 中的 `vit_tiny/vit_small/vit_base`。

## 8. ResNet 代码和预训练权重

`resnet.py` 仍然保留了完整的手写 ResNet：

- `resnet18(num_classes=...)`
- `resnet34(num_classes=...)`
- `resnet50(num_classes=...)`

对应本地 ImageNet 预训练权重位于：

```text
pretrained/
  resnet18-f37072fd.pth
  resnet34-b627a593.pth
  resnet50-0676ba61.pth
  resnet50-11ad3fa6.pth
```

`train.py` 中也仍保留了：

- `DEFAULT_PRETRAINED_WEIGHTS`
- `create_model()`
- `load_imagenet_pretrained()`

但当前这行被注释：

```python
# model = create_model(args.model, num_classes=len(class_names))
```

实际生效的是：

```python
model = ViTForImageClassification(num_classes=len(class_names))
```

因此现在的 `--model resnet18/resnet34/resnet50` 参数只会影响 checkpoint 里保存的 `model_name` 字段，不会改变实际训练网络。这一点如果后续要恢复 ResNet 训练，需要同步修改代码和 README。

## 9. 单张图片预测

`predict.py` 当前是 ResNet 推理脚本。它会从 checkpoint 读取：

```python
model_name = checkpoint["model_name"]
```

然后只支持重建：

- `resnet18`
- `resnet34`
- `resnet50`

示例命令：

```powershell
python predict.py `
  --checkpoint checkpoints/food101_resnet18/best.pt `
  --image pic/food101_val_00819_hamburger.jpg
```

注意：当前 `train.py` 实际保存的是 ViT 参数，但 checkpoint 的 `model_name` 仍来自 `--model`，默认是 `resnet18`。因此用现在的 `train.py` 训练出的 ViT checkpoint 直接交给 `predict.py`，会因为模型结构不一致而不适配。

如果要让当前 ViT checkpoint 支持单图预测，需要单独更新 `predict.py`，让它能根据 checkpoint 重建 `ViTForImageClassification`。

## 10. 常用调试命令

检查 ResNet 文件是否能独立跑：

```powershell
python resnet.py
```

检查 AlexNet 输出形状：

```powershell
python AlexNet.py
```

检查 Hugging Face ViT 分类头：

```powershell
python ViT.py
```

语法检查：

```powershell
python -m py_compile train.py resnet.py AlexNet.py ViT.py VIT_learn.py predict.py download.py
```

## 11. 当前需要特别注意的问题

- `README` 已按当前代码说明训练入口实际使用 ViT，但 `train.py` 顶部注释和部分参数说明仍写着 ResNet，这是代码注释层面的历史遗留。
- `--pretrained` / `--pretrained-path` 是 ResNet 本地权重逻辑，不是当前 ViT 的推荐使用方式。
- `predict.py` 仍是 ResNet-only，不能直接消费当前 ViT 训练得到的 checkpoint。
- `download.py` 的 mean/std 统计函数存在，但默认没有执行统计。
- `VIT_learn.py` 是学习实现，尚未接入训练脚本。
