"""
下载并检查 Hugging Face Food101 数据集。

这个脚本负责训练前的数据准备检查：
1. 通过 load_dataset("ethz/food101") 下载/读取 Food101。
2. 打印 split 信息、类别数量、类别名示例和第一条训练样本。
3. 统计 train split 自己的 RGB mean/std，输出可复制到 train.py。

注意：
- 这个脚本不进入模型训练。
- 默认会逐张扫描训练集统计 mean/std；如需快速测试，可用 --max-samples 限制样本数。
- 统计 mean/std 时只使用 train split，不使用 validation split。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torchvision import transforms


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="下载并检查 ethz/food101 数据集")
    parser.add_argument("--image-size", type=int, default=224, help="CenterCrop 后的输入边长")
    parser.add_argument(
        "--resize-margin",
        type=int,
        default=32,
        help="Resize 的额外边长；默认 224+32=256，对齐训练脚本预处理。",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="只统计前 N 张图片，主要用于快速测试；完整统计不要传。",
    )
    parser.add_argument(
        "--print-freq",
        type=int,
        default=1000,
        help="统计 mean/std 时每处理多少张图片打印一次进度。",
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


def print_dataset_summary(dataset) -> None:
    """打印数据集结构和第一条训练样本，确认数据集能正常读取。"""

    print(dataset)

    train_split = dataset["train"]
    class_names = train_split.features["label"].names
    first_sample = train_split[0]

    print(f"类别数量: {len(class_names)}")
    print(f"前 10 个类别: {class_names[:10]}")
    print(f"第一张图片尺寸: {first_sample['image'].size}")
    print(f"第一条标签 id: {first_sample['label']}")
    print(f"第一条标签名: {class_names[first_sample['label']]}")
    first_sample["image"].show()


def compute_mean_std(train_split, args: argparse.Namespace) -> dict[str, object]:
    """逐张扫描训练集，累加每个 RGB 通道的像素和与平方和。"""

    total_images = len(train_split)
    dataset = train_split

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


def print_stats_result(result: dict[str, object], output: Path | None) -> None:
    """打印并可选保存 Food101 train split 的 mean/std 统计结果。"""

    mean = tuple(round(value, 6) for value in result["mean"])
    std = tuple(round(value, 6) for value in result["std"])

    print("\n可复制到 train.py：")
    print(f"FOOD101_MEAN = {mean}")
    print(f"FOOD101_STD = {std}")

    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"\n已保存统计结果: {output}")


def main() -> None:
    args = parse_args()

    from datasets import load_dataset

    dataset = load_dataset("ethz/food101")
    print_dataset_summary(dataset)

    result = compute_mean_std(dataset["train"], args)
    print_stats_result(result, args.output)


if __name__ == "__main__":
    main()
