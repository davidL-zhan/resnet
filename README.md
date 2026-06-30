# Food101 ResNet 分类项目

本项目使用 `resnet.py` 中手写的 ResNet-18 / ResNet-34，在 Hugging Face
`ethz/food101` 数据集上完成 101 类食物图片分类。

## 1. 安装依赖

用户指定的深度学习环境是 DarkFlow：

```powershell
E:\miniconda\envs\DarkFLow\python.exe -m pip install -r requirements.txt
```

当前代码依赖：

- `torch`
- `torchvision`
- `datasets`
- `pillow`

## 2. 检查数据集

```powershell
E:\miniconda\envs\DarkFLow\python.exe download.py
```

## 3. 快速冒烟训练

先用小样本确认完整训练流程能跑通：

```powershell
E:\miniconda\envs\DarkFLow\python.exe train_food101.py `
  --subset-train 512 `
  --subset-val 128 `
  --epochs 1 `
  --batch-size 16 `
  --num-workers 0 `
  --output-dir checkpoints/food101_smoke
```

Windows 上如果 DataLoader 多进程遇到问题，先把 `--num-workers` 设为 `0`。

## 4. 正式训练

ResNet-18 示例：

```powershell
E:\miniconda\envs\DarkFLow\python.exe train_food101.py `
  --model resnet18 `
  --epochs 30 `
  --batch-size 64 `
  --lr 0.1 `
  --amp `
  --output-dir checkpoints/food101_resnet18
```

ResNet-34 示例：

```powershell
E:\miniconda\envs\DarkFLow\python.exe train_food101.py `
  --model resnet34 `
  --epochs 30 `
  --batch-size 64 `
  --lr 0.1 `
  --amp `
  --output-dir checkpoints/food101_resnet34
```

训练输出：

- `last.pt`：最后一个 epoch 的 checkpoint
- `best.pt`：验证集 top-1 最高的 checkpoint
- `metrics.jsonl`：每个 epoch 的训练/验证 loss 和 top-1

## 5. 继续训练

```powershell
E:\miniconda\envs\DarkFLow\python.exe train_food101.py `
  --resume checkpoints/food101_resnet18/last.pt `
  --epochs 60 `
  --output-dir checkpoints/food101_resnet18
```

## 6. 单张图片预测

```powershell
E:\miniconda\envs\DarkFLow\python.exe predict_food101.py `
  --checkpoint checkpoints/food101_resnet18/best.pt `
  --image path\to\food.jpg
```

## 7. 重要说明

`resnet.py` 是从零初始化的手写 ResNet，不是 ImageNet 预训练模型。  
因此 Food101 上的前几个 epoch 准确率可能比较低，这是正常现象。若目标是更强
基线，应额外加入 ImageNet 预训练微调；但本项目当前目标是验证并训练手写 ResNet。
