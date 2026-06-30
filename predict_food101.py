"""
使用 train_food101.py 保存的 checkpoint 对单张图片做 Food101 分类预测。

示例：
    python predict_food101.py --checkpoint checkpoints/food101_resnet18/best.pt --image demo.jpg

checkpoint 中保存了：
- model_name: resnet18 或 resnet34
- class_names: Food101 的 101 个类别名
- model_state: 模型参数

因此预测脚本不需要再次联网加载 Hugging Face 数据集。
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from resnet import resnet18, resnet34
from train_food101 import IMAGENET_MEAN, IMAGENET_STD


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict Food101 class with a trained ResNet")
    parser.add_argument("--checkpoint", type=Path, required=True, help="训练得到的 best.pt 或 last.pt")
    parser.add_argument("--image", type=Path, required=True, help="待分类图片路径")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="推理设备，例如 cuda、cuda:0 或 cpu。",
    )
    return parser.parse_args()


def build_eval_transform(image_size: int) -> transforms.Compose:
    """预测时使用和验证集一致的确定性预处理。"""

    return transforms.Compose(
        [
            transforms.Resize(image_size + 32),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def create_model(model_name: str, num_classes: int) -> torch.nn.Module:
    """按 checkpoint 记录的模型名重建网络结构。"""

    if model_name == "resnet18":
        return resnet18(num_classes=num_classes)
    if model_name == "resnet34":
        return resnet34(num_classes=num_classes)
    raise ValueError(f"Unsupported model in checkpoint: {model_name}")


@torch.no_grad()
def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    class_names = checkpoint["class_names"]
    model_name = checkpoint["model_name"]

    model = create_model(model_name=model_name, num_classes=len(class_names)).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    image = Image.open(args.image).convert("RGB")
    transform = build_eval_transform(args.image_size)
    input_tensor = transform(image).unsqueeze(0).to(device)

    logits = model(input_tensor)
    probabilities = torch.softmax(logits, dim=1).squeeze(0)

    topk = min(args.topk, len(class_names))
    scores, indices = torch.topk(probabilities, k=topk)

    print(f"图片: {args.image}")
    for rank, (score, index) in enumerate(zip(scores.tolist(), indices.tolist()), start=1):
        print(f"top{rank}: {class_names[index]}  prob={score:.4f}")


if __name__ == "__main__":
    main()
