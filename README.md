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

## 3. 统计 Food101 Mean/Std

`compute_food101_stats.py` 用来统计 Food101 训练集自己的 RGB 通道均值和标准差。
当前 `train_food101.py` 默认使用 ImageNet 的 `IMAGENET_MEAN/STD`，这是自然图像任务里常见的默认值；
如果希望归一化参数更贴合 Food101，可以先运行这个统计脚本，再把输出结果替换到训练脚本里。

统计脚本只读取 `ethz/food101` 的 `train` split，不使用 validation split，避免把验证集信息混进训练配置。
统计前会对每张图片执行确定性预处理：

```text
RGB 转换 -> Resize(image_size + resize_margin) -> CenterCrop(image_size) -> ToTensor()
```

默认参数是 `image_size=224`、`resize_margin=32`，也就是先 `Resize(256)` 再 `CenterCrop(224)`。
这里不用 `RandomResizedCrop`，因为随机裁剪会导致每次统计出来的 mean/std 不完全一致。

完整统计训练集：

```powershell
E:\miniconda\envs\DarkFLow\python.exe compute_food101_stats.py
```

快速测试前 512 张：

```powershell
E:\miniconda\envs\DarkFLow\python.exe compute_food101_stats.py --max-samples 512
```

保存统计结果到 JSON：

```powershell
E:\miniconda\envs\DarkFLow\python.exe compute_food101_stats.py --output food101_stats.json
```

脚本会打印可复制到 `train_food101.py` 的结果：

```python
FOOD101_MEAN = (...)
FOOD101_STD = (...)
```

替换时，把 `train_food101.py` 里的：

```python
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
```

改成统计脚本输出的 Food101 数值，并同步修改 `Normalize(...)` 使用的变量名即可。

## 4. 快速冒烟训练

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

## 5. 正式训练

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

## 6. 继续训练

```powershell
E:\miniconda\envs\DarkFLow\python.exe train_food101.py `
  --resume checkpoints/food101_resnet18/last.pt `
  --epochs 60 `
  --output-dir checkpoints/food101_resnet18
```

## 7. 单张图片预测

```powershell
E:\miniconda\envs\DarkFLow\python.exe predict_food101.py `
  --checkpoint checkpoints/food101_resnet18/best.pt `
  --image path\to\food.jpg
```

## 8. 重要说明

`resnet.py` 是从零初始化的手写 ResNet，不是 ImageNet 预训练模型。  
因此 Food101 上的前几个 epoch 准确率可能比较低，这是正常现象。若目标是更强
基线，应额外加入 ImageNet 预训练微调；但本项目当前目标是验证并训练手写 ResNet。
