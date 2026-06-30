# Food101 ResNet 分类项目

本项目使用 `resnet.py` 中手写的 ResNet-18 / ResNet-34，在 Hugging Face
`ethz/food101` 数据集上完成 101 类食物图片分类。

## 1. 数据集介绍

Food101 是一个常用的食物图像分类数据集，包含 101 个食物类别。每个类别有
1000 张图片，总计 101000 张图片。Hugging Face 版本 `ethz/food101` 提供：

- `train`：75750 张图片
- `validation`：25250 张图片
- `label`：0 到 100 的类别 id
- `image`：PIL 图片对象

本项目把 Food101 作为 101 类单标签分类任务，模型输入为 RGB 图片，输出为
101 维 logits，并使用交叉熵损失进行训练。

## 2. 创建虚拟环境并安装依赖

先用 conda 创建并激活虚拟环境：

```powershell
conda create -n food101-resnet python=3.11 -y
conda activate food101-resnet
```

然后安装项目依赖：

```powershell
pip install -r requirements.txt
```

依赖包括：

- `torch`
- `torchvision`
- `datasets`
- `pillow`
- `tqdm`

## 3. 下载并检查数据集

```powershell
python download.py
```

## 4. 统计 Food101 Mean/Std

`download.py` 会统计 Food101 训练集自己的 RGB 通道均值和标准差。
当前 `train_food101.py` 使用的是 `FOOD101_MEAN/STD`；
如果后续调整了预处理尺寸或想重新校准归一化参数，可以先运行统计命令，再把输出结果替换到训练脚本里。

统计脚本只读取 `ethz/food101` 的 `train` split，不使用 validation split，避免把验证集信息混进训练配置。
统计前会对每张图片执行确定性预处理：

```text
RGB 转换 -> Resize(image_size + resize_margin) -> CenterCrop(image_size) -> ToTensor()
```

默认参数是 `image_size=224`、`resize_margin=32`，也就是先 `Resize(256)` 再 `CenterCrop(224)`。
这里不用 `RandomResizedCrop`，因为随机裁剪会导致每次统计出来的 mean/std 不完全一致。

完整统计训练集：

```powershell
python download.py
```

快速测试前 512 张：

```powershell
python download.py --max-samples 512
```

保存统计结果到 JSON：

```powershell
python download.py --output food101_stats.json
```

脚本会打印可复制到 `train_food101.py` 的结果：

```python
FOOD101_MEAN = (...)
FOOD101_STD = (...)
```

替换时，把 `train_food101.py` 里的 `FOOD101_MEAN` 和 `FOOD101_STD`
改成 `download.py` 输出的数值即可。

## 5. 正式训练

ResNet-18 示例：

```powershell
python train_food101.py `
  --model resnet18 `
  --epochs 30 `
  --batch-size 64 `
  --lr 0.0003 `
  --amp `
  --output-dir checkpoints/food101_resnet18
```

ResNet-34 示例：

```powershell
python train_food101.py `
  --model resnet34 `
  --epochs 30 `
  --batch-size 64 `
  --lr 0.0003 `
  --amp `
  --output-dir checkpoints/food101_resnet34
```

训练输出：

- `last.pt`：最后一个 epoch 的 checkpoint
- `best.pt`：验证集 top-1 最高的 checkpoint
- `metrics.jsonl`：每个 epoch 的训练/验证 loss 和 top-1

Windows 上如果 DataLoader 多进程遇到问题，先把 `--num-workers` 设为 `0`。

## 6. 继续训练

```powershell
python train_food101.py `
  --resume checkpoints/food101_resnet18/last.pt `
  --epochs 60 `
  --output-dir checkpoints/food101_resnet18
```

## 7. 单张图片预测

```powershell
python predict_food101.py `
  --checkpoint checkpoints/food101_resnet18/best.pt `
  --image path\to\food.jpg
```

## 8. 重要说明

`resnet.py` 是从零初始化的手写 ResNet，不是 ImageNet 预训练模型。
因此 Food101 上的前几个 epoch 准确率可能比较低，这是正常现象。若目标是更强
基线，应额外加入 ImageNet 预训练微调；但本项目当前目标是验证并训练手写 ResNet。
