"""
检查 Hugging Face Food101 数据集是否可以正常加载。

这个脚本只负责“数据集连通性检查”，不负责训练模型：
1. 加载 ethz/food101 的数据集描述和 split 信息。
2. 打印类别数量、类别名示例、第一条样本的图片尺寸和标签。
3. 真正训练入口在 train_food101.py。
"""

from __future__ import annotations

import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="检查 ethz/food101 数据集加载状态")
    return parser.parse_args()


def main() -> None:
    parse_args()

    from datasets import load_dataset

    dataset = load_dataset("ethz/food101")

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


if __name__ == "__main__":
    main()
