"""
统计 Food101 训练集自己的 RGB mean/std。

为什么单独写这个脚本：
- train_food101.py 里当前使用的是 ImageNet 的 mean/std，这是自然图像分类中常见默认值。
- 如果想更贴合 Food101，从训练集本身统计 mean/std 更严格。
- 统计时只使用 train split，不能把 validation split 混进去，否则会引入验证集信息。

统计口径：
- 读取 ethz/food101 的 train split。
- 每张图统一转成 RGB。
- 使用和验证/推理一致的确定性预处理：Resize(image_size + resize_margin)
  再 CenterCrop(image_size)，最后 ToTensor。
- 不使用 RandomResizedCrop，因为随机增强会让每次统计结果不同。

运行示例：
    E:\\miniconda\\envs\\DarkFLow\\python.exe compute_food101_stats.py

快速测试前 512 张：
    E:\\miniconda\\envs\\DarkFLow\\python.exe compute_food101_stats.py --max-samples 512
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torchvision import transforms


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute Food101 train-set RGB mean/std")
    parser.add_argument("--image-size", type=int, default=224, help="CenterCrop 后的输入边长")
    parser.add_argument(
        "--resize-margin",
        type=int,
        default=32,
        help="Resize 的额外边长；默认 224+32=256，对齐当前训练脚本的预处理。",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="只统计前 N 张图片，主要用于快速测试；正式统计不要传。",
    )
    parser.add_argument(
        "--print-freq",
        type=int,
        default=1000,
        help="每处理多少张图片打印一次进度。",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="可选 JSON 输出路径，用于保存 mean/std 和统计配置。",
    )
    return parser.parse_args()


def build_stat_transform(image_size: int, resize_margin: int) -> transforms.Compose:
    """
    构造统计 mean/std 时使用的确定性预处理。

    ToTensor 会把 PIL 图片从 [0, 255] 转成 float tensor [0, 1]，
    mean/std 也是在 [0, 1] 的数值空间里统计。
    """

    return transforms.Compose(
        [
            transforms.Resize(image_size + resize_margin),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
        ]
    )


def compute_mean_std(args: argparse.Namespace) -> dict[str, object]:
    """逐张扫描训练集，累加每个 RGB 通道的像素和与平方和。"""

    from datasets import load_dataset

    dataset = load_dataset("ethz/food101", split="train")
    total_images = len(dataset)

    if args.max_samples is not None:
        total_images = min(args.max_samples, total_images)
        dataset = dataset.select(range(total_images))

    transform = build_stat_transform(
        image_size=args.image_size,
        resize_margin=args.resize_margin,
    )

    channel_sum = torch.zeros(3, dtype=torch.float64)
    channel_squared_sum = torch.zeros(3, dtype=torch.float64)
    pixel_count = 0

    for index, sample in enumerate(dataset, start=1):
        image = sample["image"].convert("RGB")
        tensor = transform(image).to(dtype=torch.float64)

        # tensor shape: [3, H, W]。对 H/W 维求和，保留 RGB 三个通道。
        channel_sum += tensor.sum(dim=(1, 2))
        channel_squared_sum += (tensor * tensor).sum(dim=(1, 2))
        pixel_count += tensor.shape[1] * tensor.shape[2]

        if index % args.print_freq == 0 or index == total_images:
            print(f"processed {index}/{total_images} images")

    mean = channel_sum / pixel_count
    variance = channel_squared_sum / pixel_count - mean * mean
    std = torch.sqrt(torch.clamp(variance, min=0.0))

    return {
        "dataset": "ethz/food101",
        "split": "train",
        "num_images": total_images,
        "image_size": args.image_size,
        "resize_margin": args.resize_margin,
        "mean": [float(value) for value in mean.tolist()],
        "std": [float(value) for value in std.tolist()],
    }


def main() -> None:
    args = parse_args()
    result = compute_mean_std(args)

    mean = tuple(round(value, 6) for value in result["mean"])
    std = tuple(round(value, 6) for value in result["std"])

    print("\n可复制到 train_food101.py：")
    print(f"FOOD101_MEAN = {mean}")
    print(f"FOOD101_STD = {std}")

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"\n已保存统计结果: {args.output}")


if __name__ == "__main__":
    main()
