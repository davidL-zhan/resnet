"""
使用本项目的手写 ResNet 在 Hugging Face Food101 数据集上训练分类模型。

核心流程：
1. load_dataset("ethz/food101") 下载/读取 Food101。
2. 对 PIL 图片做训练增强或验证预处理。
3. 调用 resnet.py 中的 resnet18 / resnet34 / resnet50，设置 num_classes=101。
4. 用交叉熵训练多分类模型，并保存 last.pt 和 best.pt。

Food101 是 101 类食物图片分类数据集。这里默认从零训练本项目的 ResNet；
传入 --pretrained 或 --pretrained-path 时，会加载 pretrained/ 下的本地
ImageNet-1K 预训练权重。Reviewer #2 视角下需要注意：
从零训练 ResNet-18 在 Food101 上收敛会明显慢于预训练微调，完整训练通常需要
较多 epoch；如果只跑几个 epoch，验证准确率低是预期现象，不代表代码错误。
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from functools import partial
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm.auto import tqdm

from resnet import resnet18, resnet34, resnet50
from VIT import vit_base_patch16_224
import warnings

warnings.filterwarnings("ignore", message="Truncated File Read")
FOOD101_MEAN = (0.557, 0.442, 0.327)
FOOD101_STD = (0.259, 0.263, 0.265)


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

DEFAULT_PRETRAINED_WEIGHTS = {
    "resnet18": Path("pretrained/resnet18-f37072fd.pth"),
    "resnet34": Path("pretrained/resnet34-b627a593.pth"),
    "resnet50": Path("pretrained/resnet50-11ad3fa6.pth"),
}


@dataclass
class EpochResult:
    """保存一个 epoch 的聚合指标，方便打印和写入日志。"""

    loss: float
    top1: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train ResNet on ethz/food101")

    # 数据相关参数。
    parser.add_argument(
        "--image-size", type=int, default=224, help="输入图片裁剪后的边长"
    )
    # 模型和优化参数。
    parser.add_argument(
        "--model", choices=["resnet18", "resnet34", "resnet50"], default="resnet18"
    )
    parser.add_argument(
        "--pretrained",
        action="store_true",
        help="从 pretrained/ 加载 ImageNet-1K 预训练权重，跳过最后的 fc 分类层。",
    )
    parser.add_argument(
        "--pretrained-path",
        type=Path,
        default=None,
        help="手动指定本地预训练权重路径；不传则按 --model 自动选择。",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument(
        "--amp",
        action="store_true",
        help="CUDA 上启用自动混合精度，可降低显存占用并提升速度。",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="训练设备，例如 cuda、cuda:0 或 cpu。",
    )
    parser.add_argument("--seed", type=int, default=42)

    # 输出与恢复训练。
    parser.add_argument(
        "--output-dir", type=Path, default=Path("checkpoints/food101_resnet18")
    )
    parser.add_argument(
        "--resume", type=Path, default=None, help="从某个 checkpoint 继续训练"
    )
    parser.add_argument(
        "--print-freq", type=int, default=50, help="每多少个 step 刷新一次 tqdm 指标"
    )

    return parser.parse_args()


def set_seed(seed: int) -> None:
    """固定随机种子，让小规模调试的结果更容易复现。"""

    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # benchmark=True 会根据输入尺寸自动选择更快卷积实现。
    # 这里输入尺寸固定为 224x224，因此打开它通常更快。
    torch.backends.cudnn.benchmark = True


def build_transforms(
    image_size: int,
    use_imagenet_norm: bool,
) -> tuple[transforms.Compose, transforms.Compose]:
    """
    构造训练和验证预处理。

    训练阶段使用随机裁剪和水平翻转，给模型制造更多视角；
    验证阶段使用确定性的 resize + center crop，保证指标稳定。
    """

    mean = IMAGENET_MEAN if use_imagenet_norm else FOOD101_MEAN
    std = IMAGENET_STD if use_imagenet_norm else FOOD101_STD
    norm_name = "ImageNet" if use_imagenet_norm else "Food101"
    print(f"输入归一化: {norm_name} mean/std")

    train_transform = transforms.Compose(
        [
            transforms.Resize(image_size + 32),
            transforms.RandomResizedCrop(image_size, scale=(0.6, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )

    eval_transform = transforms.Compose(
        [
            transforms.Resize(image_size + 32),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )

    return train_transform, eval_transform


def apply_train_transform(
    batch: dict[str, Any],
    transform: transforms.Compose,
) -> dict[str, Any]:
    """
    对训练 batch 应用图片增强。

    这个函数必须放在文件顶层，因为 Windows 下 DataLoader(num_workers>0)
    使用 spawn 启动子进程，需要 pickle dataset transform。局部函数不能 pickle。
    """

    # Hugging Face Dataset 的 with_transform 会把样本按 batch 形式传进来，
    # 因此这里要对 batch["image"] 中的每张 PIL 图片逐一做 transform。
    batch["pixel_values"] = [transform(ensure_rgb(image)) for image in batch["image"]]
    return batch


def apply_eval_transform(
    batch: dict[str, Any],
    transform: transforms.Compose,
) -> dict[str, Any]:
    """对验证 batch 应用确定性预处理。"""

    batch["pixel_values"] = [transform(ensure_rgb(image)) for image in batch["image"]]
    return batch


def load_food101(args: argparse.Namespace):
    """
    加载 Food101，并返回 train/validation split 和类别名。

    datasets 的样本图片是 PIL.Image，不能直接送进 PyTorch 模型；
    因此这里使用 with_transform 延迟执行 torchvision transform。
    延迟执行的好处是：不用一次性把所有图片转成 tensor 写入内存。
    """

    # datasets 放在函数里导入，让脚本启动时不必立刻加载该依赖。
    from datasets import load_dataset

    raw_dataset = load_dataset("ethz/food101")

    class_names = raw_dataset["train"].features["label"].names
    train_dataset = raw_dataset["train"]
    val_dataset = raw_dataset["validation"]

    train_transform, eval_transform = build_transforms(
        image_size=args.image_size,
        use_imagenet_norm=args.pretrained,
    )

    return (
        train_dataset.with_transform(
            partial(apply_train_transform, transform=train_transform)
        ),
        val_dataset.with_transform(
            partial(apply_eval_transform, transform=eval_transform)
        ),
        class_names,
    )


def ensure_rgb(image: Image.Image) -> Image.Image:
    """
    Food101 通常是 RGB 图片，但这里仍统一转换一次。

    这样可以避免少数灰度图、带 alpha 通道图片导致模型输入通道数不等于 3。
    """

    return image.convert("RGB")


def collate_food101(batch: list[dict[str, Any]]) -> tuple[torch.Tensor, torch.Tensor]:
    """
    将 Hugging Face Dataset 的样本列表整理成 PyTorch batch。

    with_transform 后每条样本里有 pixel_values 和 label；
    原始 image 是 PIL 对象，默认 DataLoader 无法直接拼接，所以 collate 时只取
    模型真正需要的两个字段。
    """

    images = torch.stack([sample["pixel_values"] for sample in batch], dim=0)
    labels = torch.tensor([sample["label"] for sample in batch], dtype=torch.long)
    return images, labels


def build_dataloaders(
    train_dataset,
    val_dataset,
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> tuple[DataLoader, DataLoader]:
    """创建训练和验证 DataLoader。"""

    pin_memory = device.type == "cuda"

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_food101,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_food101,
    )

    return train_loader, val_loader


def create_model(model_name: str, num_classes: int) -> nn.Module:
    """根据命令行参数创建本项目 resnet.py 里的模型。"""

    if model_name == "resnet18":
        model = resnet18(num_classes=num_classes)
    elif model_name == "resnet34":
        model = resnet34(num_classes=num_classes)
    elif model_name == "resnet50":
        model = resnet50(num_classes=num_classes)
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    total_params = sum(param.numel() for param in model.parameters())
    trainable_params = sum(
        param.numel() for param in model.parameters() if param.requires_grad
    )
    print(
        f"使用模型: {model_name} | "
        f"总参数量: {total_params:,} | "
        f"可训练参数量: {trainable_params:,}"
    )
    return model


def resolve_pretrained_path(model_name: str, pretrained_path: Path | None) -> Path:
    """根据模型名解析本地 ImageNet 预训练权重路径。"""

    if pretrained_path is not None:
        return pretrained_path

    if model_name not in DEFAULT_PRETRAINED_WEIGHTS:
        raise ValueError(f"当前模型没有默认预训练权重路径: {model_name}")

    return DEFAULT_PRETRAINED_WEIGHTS[model_name]


def load_imagenet_pretrained(
    model: nn.Module,
    model_name: str,
    pretrained_path: Path | None,
) -> None:
    """
    给本项目手写 ResNet 加载本地 ImageNet-1K 预训练权重。

    这里先用 resnet.py 构建 Food101 的 101 类模型，再加载本地 .pth。
    ImageNet 是 1000 类，Food101 是 101 类，因此 fc.weight / fc.bias
    形状不匹配，会被自动跳过并保留当前随机初始化的 Food101 分类头。
    """

    weight_path = resolve_pretrained_path(model_name, pretrained_path)
    if not weight_path.is_file():
        raise FileNotFoundError(
            f"找不到预训练权重: {weight_path}。请先把权重下载到 pretrained/，"
            "或用 --pretrained-path 指定正确路径。"
        )

    checkpoint = torch.load(weight_path, map_location="cpu")
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        pretrained_state = checkpoint["state_dict"]
    else:
        pretrained_state = checkpoint

    model_state = model.state_dict()
    compatible_state = {}
    skipped_keys = []
    unexpected_keys = []

    for key, value in pretrained_state.items():
        clean_key = key.removeprefix("module.")

        if clean_key not in model_state:
            unexpected_keys.append(clean_key)
            continue

        if model_state[clean_key].shape != value.shape:
            skipped_keys.append(
                f"{clean_key}: pretrained={tuple(value.shape)}, model={tuple(model_state[clean_key].shape)}"
            )
            continue

        compatible_state[clean_key] = value

    missing_keys, load_unexpected_keys = model.load_state_dict(
        compatible_state, strict=False
    )

    print(f"已加载 ImageNet 预训练权重: {weight_path}")
    print(f"成功加载参数数量: {len(compatible_state)}")
    if skipped_keys:
        print("因形状不匹配跳过的参数:")
        for item in skipped_keys:
            print(f"  - {item}")
    if unexpected_keys or load_unexpected_keys:
        print(
            f"权重文件中未使用的参数数量: {len(unexpected_keys) + len(load_unexpected_keys)}"
        )
    if missing_keys:
        print("模型中未从预训练权重加载的参数:")
        for key in missing_keys:
            print(f"  - {key}")


def accuracy_top1(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """计算当前 batch 的 top-1 accuracy。"""

    predictions = logits.argmax(dim=1)
    correct = (predictions == labels).sum().item()
    return correct / labels.numel()


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    epoch: int,
    print_freq: int,
) -> EpochResult:
    """执行一个训练 epoch。"""

    model.train()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    update_freq = max(print_freq, 1)

    progress_bar = tqdm(
        loader,
        total=len(loader),
        desc=f"epoch {epoch} train",
        dynamic_ncols=True,
    )

    for step, (images, labels) in enumerate(progress_bar, start=1):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        # autocast 只在 CUDA AMP 开启时生效；CPU 或未传 --amp 时就是普通 FP32。
        with torch.amp.autocast("cuda", enabled=scaler.is_enabled()):
            logits = model(images)
            loss = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (logits.argmax(dim=1) == labels).sum().item()
        total_samples += batch_size

        if step % update_freq == 0 or step == len(loader):
            avg_loss = total_loss / total_samples
            avg_acc = total_correct / total_samples
            progress_bar.set_postfix(
                train_loss=f"{avg_loss:.4f}",
                train_top1=f"{avg_acc:.4f}",
            )

    return EpochResult(
        loss=total_loss / total_samples, top1=total_correct / total_samples
    )


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> EpochResult:
    """在验证集上评估模型。"""

    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(images)
        loss = criterion(logits, labels)

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (logits.argmax(dim=1) == labels).sum().item()
        total_samples += batch_size

    return EpochResult(
        loss=total_loss / total_samples, top1=total_correct / total_samples
    )


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    epoch: int,
    best_top1: float,
    class_names: list[str],
    args: argparse.Namespace,
) -> None:
    """保存训练状态，既支持推理，也支持 resume。"""

    path.parent.mkdir(parents=True, exist_ok=True)

    serializable_args = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }

    checkpoint = {
        "epoch": epoch,
        "model_name": args.model,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "best_top1": best_top1,
        "num_classes": len(class_names),
        "class_names": class_names,
        "args": serializable_args,
    }
    torch.save(checkpoint, path)


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    """将每个 epoch 的指标追加写入 jsonl，方便后续画曲线或查日志。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_checkpoint_if_needed(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    resume_path: Path | None,
    device: torch.device,
) -> tuple[int, float]:
    """如果传入 --resume，则恢复模型、优化器、scheduler。"""

    if resume_path is None:
        return 1, 0.0

    checkpoint = torch.load(resume_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    scheduler.load_state_dict(checkpoint["scheduler_state"])

    start_epoch = int(checkpoint["epoch"]) + 1
    best_top1 = float(checkpoint.get("best_top1", 0.0))

    print(
        f"从 {resume_path} 恢复训练：start_epoch={start_epoch}, best_top1={best_top1:.4f}"
    )
    return start_epoch, best_top1


def main() -> None:
    args = parse_args()
    if args.pretrained_path is not None:
        args.pretrained = True

    set_seed(args.seed)

    train_dataset, val_dataset, class_names = load_food101(args)

    print(f"训练样本数: {len(train_dataset)}")
    print(f"验证样本数: {len(val_dataset)}")
    print(f"类别数量: {len(class_names)}")
    print(f"前 10 个类别: {class_names[:10]}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader = build_dataloaders(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=device,
    )

    # model = create_model(args.model, num_classes=len(class_names))
    model = vit_base_patch16_224(101, 3)
    if args.pretrained:
        load_imagenet_pretrained(
            model=model,
            model_name=args.model,
            pretrained_path=args.pretrained_path,
        )
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()

    # AdamW 对学习率更敏感，默认使用 3e-4，不沿用 SGD 常见的 0.1。
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")

    start_epoch, best_top1 = load_checkpoint_if_needed(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        resume_path=args.resume,
        device=device,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = args.output_dir / "metrics.jsonl"

    print(f"输出目录: {args.output_dir}")
    print(f"训练设备: {device}")
    print(f"AMP: {scaler.is_enabled()}")

    for epoch in range(start_epoch, args.epochs + 1):
        train_result = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            epoch=epoch,
            print_freq=args.print_freq,
        )
        val_result = evaluate(
            model=model, loader=val_loader, criterion=criterion, device=device
        )
        scheduler.step()

        is_best = val_result.top1 > best_top1
        best_top1 = max(best_top1, val_result.top1)

        record = {
            "epoch": epoch,
            "lr": scheduler.get_last_lr()[0],
            "train": asdict(train_result),
            "validation": asdict(val_result),
            "best_top1": best_top1,
        }
        append_jsonl(metrics_path, record)

        print(
            f"epoch={epoch} "
            f"train_loss={train_result.loss:.4f} train_top1={train_result.top1:.4f} "
            f"val_loss={val_result.loss:.4f} val_top1={val_result.top1:.4f} "
            f"best_top1={best_top1:.4f}"
        )

        save_checkpoint(
            path=args.output_dir / "last.pt",
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            best_top1=best_top1,
            class_names=class_names,
            args=args,
        )

        if is_best:
            save_checkpoint(
                path=args.output_dir / "best.pt",
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                best_top1=best_top1,
                class_names=class_names,
                args=args,
            )


if __name__ == "__main__":
    main()
