from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import torch
from PIL import Image
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TaskID,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import InterpolationMode

from orchestration.train_env import (
    apply_seed,
    create_console,
    env_float,
    env_int,
    env_path,
    env_str,
    load_transform_toggles,
    maybe_load_checkpoint,
    prepare_training_environment,
    require_num_classes,
    save_best_checkpoint,
    save_latest_checkpoint,
)

# ---------------------------- Config --------------------------------- #

DEFAULT_EPOCHS: int = 50
DEFAULT_BATCH_SIZE: int = 64
DEFAULT_IMG_SIZE: int = 224
DEFAULT_NUM_WORKERS: int = 4

# 学习率和权重衰减
LR: float = 1e-3
WD: float = 1e-4

# 早停
DEFAULT_PATIENCE: int = 10

# 输出文件名
BEST_WEIGHTS_NAME: str = "SimpleCNNModel.pth"
BEST_CKPT_NAME: str = "best.ckpt"
LATEST_CKPT_NAME: str = "latest.ckpt"

# --------------------------------------------------------------------- #

console = create_console()


def _ensure_rgb(image: Image.Image) -> Image.Image:
    """将灰度图像转换为 RGB。"""
    if getattr(image, "mode", "RGB") != "RGB":
        return image.convert("RGB")
    return image


@dataclass(frozen=True)
class EvalResult:
    """评估指标容器。"""

    acc: float
    loss: float
    total: int
    correct: int


class SimpleCNN(nn.Module):
    """用于二分类的简单 CNN 架构。
    
    该网络采用经典的卷积神经网络设计，包含四个卷积块用于特征提取，
    然后通过全局平均池化和全连接层进行分类。
    
    架构特点：
    - 逐层增加通道数（32 -> 64 -> 128 -> 256），提取更复杂的特征
    - 每个卷积块后使用最大池化进行下采样，减少计算量
    - 使用批归一化加速训练并提高稳定性
    - 分类器使用 Dropout 防止过拟合
    
    输入输出：
    - 输入：形状为 (B, 3, H, W) 的图像张量，其中 B 是批次大小，H 和 W 是图像高度和宽度
    - 输出：形状为 (B, num_classes) 的 logits 张量
    """

    def __init__(self, num_classes: int = 2) -> None:
        """初始化 SimpleCNN 模型。
        
        参数:
            num_classes: 分类类别数，默认为 2（二分类任务）
        """
        super().__init__()
        
        # 特征提取层：四个卷积块，逐层提取更高级的特征
        self.features = nn.Sequential(
            # 第一个卷积块：从 RGB 输入提取基础特征
            # 输入: (B, 3, H, W) -> 输出: (B, 32, H/2, W/2)
            # Conv2d: 3 通道输入 -> 32 通道输出，3x3 卷积核，padding=1 保持空间尺寸
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            # 批归一化：对 32 个通道进行归一化，加速训练并提高稳定性
            nn.BatchNorm2d(32),
            # ReLU 激活函数：引入非线性，inplace=True 节省内存
            nn.ReLU(inplace=True),
            # 最大池化：2x2 窗口，步长 2，将空间尺寸减半 (H, W) -> (H/2, W/2)
            nn.MaxPool2d(2, 2),
            
            # 第二个卷积块：提取中级特征
            # 输入: (B, 32, H/2, W/2) -> 输出: (B, 64, H/4, W/4)
            # 通道数翻倍，提取更复杂的特征模式
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # 第三个卷积块：提取高级特征
            # 输入: (B, 64, H/4, W/4) -> 输出: (B, 128, H/8, W/8)
            # 进一步增加通道数，捕获更抽象的特征
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # 第四个卷积块：提取深层特征
            # 输入: (B, 128, H/8, W/8) -> 输出: (B, 256, H/16, W/16)
            # 最高通道数，提取最抽象的特征表示
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        
        # 全局平均池化：将特征图池化为 1x1，输出 (B, 256, 1, 1)
        # AdaptiveAvgPool2d 可以处理任意输入尺寸，输出固定尺寸
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # 分类器：将特征向量映射到类别 logits
        self.classifier = nn.Sequential(
            # Dropout: 训练时以 0.5 的概率随机丢弃神经元，防止过拟合
            nn.Dropout(0.5),
            # 第一个全连接层：256 维特征 -> 128 维隐藏层
            nn.Linear(256, 128),
            # ReLU 激活函数
            nn.ReLU(inplace=True),
            # 再次 Dropout，进一步正则化
            nn.Dropout(0.5),
            # 第二个全连接层：128 维 -> num_classes 维，输出类别 logits
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播。
        
        参数:
            x: 输入图像张量，形状为 (B, 3, H, W)
            
        返回:
            输出 logits 张量，形状为 (B, num_classes)
        """
        # 步骤 1: 通过特征提取层，提取多尺度特征
        # (B, 3, H, W) -> (B, 256, H/16, W/16)
        x = self.features(x)
        
        # 步骤 2: 全局平均池化，将空间维度压缩为 1x1
        # (B, 256, H/16, W/16) -> (B, 256, 1, 1)
        x = self.avgpool(x)
        
        # 步骤 3: 展平特征图，将 (B, 256, 1, 1) 展平为 (B, 256)
        # 从第 1 维开始展平（保留批次维度）
        x = torch.flatten(x, 1)
        
        # 步骤 4: 通过分类器，将特征向量映射到类别 logits
        # (B, 256) -> (B, num_classes)
        x = self.classifier(x)
        
        return x


def get_loaders(
    data_root: Path,
    train_split: str,
    val_split: str,
    img_size: int,
    batch_size: int,
    num_workers: int,
    *,
    expected_classes: int | None = None,
) -> tuple[DataLoader, DataLoader]:
    """构建训练/验证数据加载器。"""
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )

    defaults = {
        "ensure_rgb": True,
        "train_resize": True,
        "train_random_resized_crop": True,
        "train_random_horizontal_flip": True,
        "train_color_jitter": True,
        "train_to_tensor": True,
        "train_normalize": True,
        "val_resize": True,
        "val_center_crop": True,
        "val_to_tensor": True,
        "val_normalize": True,
    }
    toggles = load_transform_toggles(
        defaults,
        required=("train_to_tensor", "train_normalize", "val_to_tensor", "val_normalize"),
    )

    train_ops: list[object] = []
    if toggles.get("ensure_rgb", True):
        train_ops.append(transforms.Lambda(_ensure_rgb))
    if toggles.get("train_random_resized_crop", True):
        train_ops.append(transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)))
    elif toggles.get("train_resize", True):
        train_ops.append(transforms.Resize(img_size, interpolation=InterpolationMode.BILINEAR))
    if toggles.get("train_random_horizontal_flip", True):
        train_ops.append(transforms.RandomHorizontalFlip())
    if toggles.get("train_color_jitter", False):
        train_ops.append(transforms.ColorJitter(0.2, 0.2, 0.2, 0.05))
    if toggles.get("train_to_tensor", True):
        train_ops.append(transforms.ToTensor())
    if toggles.get("train_normalize", True):
        train_ops.append(normalize)

    val_ops: list[object] = []
    if toggles.get("ensure_rgb", True):
        val_ops.append(transforms.Lambda(_ensure_rgb))
    if toggles.get("val_resize", True):
        val_ops.append(transforms.Resize(img_size + 32, interpolation=InterpolationMode.BILINEAR))
    if toggles.get("val_center_crop", True):
        val_ops.append(transforms.CenterCrop(img_size))
    if toggles.get("val_to_tensor", True):
        val_ops.append(transforms.ToTensor())
    if toggles.get("val_normalize", True):
        val_ops.append(normalize)

    train_t = transforms.Compose(train_ops)
    val_t = transforms.Compose(val_ops)

    train_ds = datasets.ImageFolder(data_root / train_split, transform=train_t)
    if expected_classes is not None:
        require_num_classes(train_ds, expected_classes, split=train_split)
    val_ds = datasets.ImageFolder(data_root / val_split, transform=val_t)

    train_dl = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        **({"prefetch_factor": 2} if num_workers > 0 else {}),
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        **({"prefetch_factor": 2} if num_workers > 0 else {}),
    )
    return train_dl, val_dl


def evaluate(
    model: nn.Module,
    dl: DataLoader,
    device: str,
    criterion: nn.Module,
) -> EvalResult:
    """计算 top-1 准确率和平均损失。"""
    model.eval()
    correct = 0
    total = 0
    loss_sum = 0.0
    with torch.inference_mode():
        for batch_x, batch_y in dl:
            inputs = batch_x.to(device, non_blocking=True)
            targets = batch_y.to(device, non_blocking=True)
            logits = model(inputs)
            loss = criterion(logits, targets)
            pred = logits.argmax(1)
            correct += (pred == targets).sum().item()
            total += targets.numel()
            loss_sum += float(loss.item()) * targets.size(0)
    acc = correct / max(1, total)
    mean_loss = loss_sum / max(1, total)
    return EvalResult(acc=acc, loss=mean_loss, total=total, correct=correct)


def train_one_epoch(
    model: nn.Module,
    dl: DataLoader,
    opt: optim.Optimizer,
    scaler: torch.amp.GradScaler,
    criterion: nn.Module,
    device: str,
    *,
    use_cuda_amp: bool,
    progress: Progress,
    task: TaskID,
) -> float:
    """单轮训练循环。"""
    model.train()
    start = perf_counter()
    opt.zero_grad(set_to_none=True)

    loss_sum = 0.0
    seen_total = 0

    for i, (batch_x, batch_y) in enumerate(dl, 1):
        inputs = batch_x.to(device, non_blocking=True)
        targets = batch_y.to(device, non_blocking=True)

        with torch.amp.autocast(device_type="cuda", enabled=use_cuda_amp):
            logits = model(inputs)
            loss = criterion(logits, targets)

        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)

        bsz = targets.size(0)
        seen_total += bsz
        loss_sum += float(loss.item()) * bsz

        elapsed = perf_counter() - start
        seen = min(i * dl.batch_size, len(dl.dataset))
        ips = seen / max(1e-6, elapsed)
        progress.update(
            task,
            advance=1,
            description=f"train | loss={loss.item():.4f} | {ips:.0f} img/s",
        )

    return loss_sum / max(1, seen_total)


def main() -> None:
    """入口函数：数据、模型、训练循环、早停、保存最佳模型。"""
    env = prepare_training_environment(
        weights_name=BEST_WEIGHTS_NAME,
        best_checkpoint_name=BEST_CKPT_NAME,
        latest_checkpoint_name=LATEST_CKPT_NAME,
    )
    apply_seed(env.seed)

    data_root = env_path("DATA_ROOT", Path("/tmp"))
    train_split = env_str("TRAIN_SPLIT", "train")
    val_split = env_str("VAL_SPLIT", "val")
    batch_size = env_int("BATCH_SIZE", DEFAULT_BATCH_SIZE)
    epochs = env_int("EPOCHS", DEFAULT_EPOCHS)
    img_size = env_int("IMG_SIZE", DEFAULT_IMG_SIZE)
    num_workers = env_int("NUM_WORKERS", DEFAULT_NUM_WORKERS)
    num_classes = env_int("NUM_CLASSES", 2)
    lr = env_float("LR", LR)
    wd = env_float("WEIGHT_DECAY", WD)
    early_stop_patience = env_int("EARLY_STOP_PATIENCE", DEFAULT_PATIENCE)

    use_cuda = torch.cuda.is_available()
    device = "cuda" if use_cuda else "cpu"
    if env.device_override:
        requested = env.device_override
        if requested.startswith("cuda") and not torch.cuda.is_available():
            console.print(
                "[bold yellow]⚠️  Requested CUDA device not available[/]; falling back to CPU",
            )
            device = "cpu"
            use_cuda = False
        else:
            device = requested
            use_cuda = requested.startswith("cuda")
    torch.backends.cudnn.benchmark = use_cuda and env.seed is None

    if not (data_root / train_split).exists() or not (data_root / val_split).exists():
        console.print(f"[bold red]Dataset not found under[/] {data_root}")
        console.print(
            f"Expected: {data_root}/{train_split}/<class> and {data_root}/{val_split}/<class>",
        )
        raise SystemExit(1)

    try:
        train_dl, val_dl = get_loaders(
            data_root,
            train_split,
            val_split,
            img_size,
            batch_size,
            num_workers,
            expected_classes=num_classes,
        )
    except ValueError as exc:
        console.print(
            "[bold red]Class configuration mismatch[/]",
            f"→ {exc}",
        )
        raise SystemExit(1) from exc

    console.print(
        f"[bold]Data[/]: train={len(train_dl.dataset)} | val={len(val_dl.dataset)} | "
        f"bs={batch_size} | steps/epoch={len(train_dl)}",
    )

    model = SimpleCNN(num_classes=num_classes)
    model = model.to(device)

    criterion: nn.Module = nn.CrossEntropyLoss(label_smoothing=0.1)
    scaler = torch.amp.GradScaler(enabled=use_cuda)

    progress = Progress(
        TextColumn("[bold blue]{task.description}"),
        BarColumn(bar_width=None),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        TextColumn("{task.fields[extra]}"),
        console=console,
        transient=False,
    )

    best_val_acc = -1.0
    best_epoch = -1
    epochs_no_improve = 0

    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs))

    start_epoch = 0
    resume_state = maybe_load_checkpoint(
        env,
        model=model,
        optimizer=opt,
        scheduler=scheduler,
    )
    if resume_state is not None:
        start_epoch = int(resume_state.get("epoch", 0))
        best_val_acc = float(resume_state.get("best_val_acc", best_val_acc))
        best_epoch = int(resume_state.get("best_epoch", best_epoch))
        epochs_no_improve = max(0, start_epoch - best_epoch)
        console.print(
            f"[bold green]Resumed[/] from epoch {start_epoch} using {env.resume_checkpoint}",
        )

    with progress:
        for epoch in range(start_epoch + 1, epochs + 1):
            task = progress.add_task(f"epoch {epoch}", total=len(train_dl), extra="")
            train_loss = train_one_epoch(
                model=model,
                dl=train_dl,
                opt=opt,
                scaler=scaler,
                criterion=criterion,
                device=device,
                use_cuda_amp=use_cuda,
                progress=progress,
                task=task,
            )
            scheduler.step()

            res = evaluate(model, val_dl, device, criterion)
            console.print(
                f"[bold cyan]epoch {epoch}[/] | train_loss={train_loss:.4f} "
                f"| val_loss={res.loss:.4f} | val_acc={res.acc:.4f} "
                f"({res.correct}/{res.total}) | lr={scheduler.get_last_lr()[0]:.2e}",
            )

            improved = res.acc > best_val_acc + 1e-4
            if improved:
                best_val_acc = res.acc
                best_epoch = epoch
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            state = save_latest_checkpoint(
                env,
                model=model,
                optimizer=opt,
                scheduler=scheduler,
                epoch=epoch,
                best_val_acc=best_val_acc,
                best_epoch=best_epoch,
                extra={},
            )

            if improved:
                save_best_checkpoint(env, state)
                console.print(
                    f"[bold green]new best[/] val_acc={best_val_acc:.4f} "
                    f"(epoch {best_epoch}) → saved {env.best_weights_path.name}",
                )
            elif epochs_no_improve >= early_stop_patience:
                console.print(
                    f"[bold yellow]Early stopping[/]: no improvement for {early_stop_patience} epoch(s). "
                    f"Best at epoch {best_epoch} with val_acc={best_val_acc:.4f}.",
                )
                break

    console.print(f"[bold green]Best weights saved →[/] {env.best_weights_path.resolve()}")
    console.print(
        f"[bold green]Best checkpoint saved →[/] {env.best_checkpoint_path.resolve()}",
    )


if __name__ == "__main__":
    main()

