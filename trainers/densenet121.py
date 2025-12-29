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

# ---------------------------- Model Definition ---------------------- #


class DenseLayer(nn.Module):
    """DenseNet 中使用的密集层。
    
    每个密集层产生 'growth_rate' 个新特征图，并将它们与所有之前的特征图连接。
    这创建了密集连接，其中每一层都接收来自所有前序层的特征图。
    
    该层使用瓶颈设计: 1x1 卷积（瓶颈）-> 3x3 卷积（特征提取）。
    """

    def __init__(self, num_input_features: int, growth_rate: int) -> None:
        """初始化密集层。
        
        参数:
            num_input_features: 输入特征图数量（从前序层累积）
            growth_rate: 要产生的新特征图数量（通常为 32）
        """
        super().__init__()
        # 瓶颈：1x1 卷积以减少计算量
        self.bn1 = nn.BatchNorm2d(num_input_features)
        self.relu1 = nn.ReLU(inplace=True)
        # 扩展到 4*growth_rate 用于瓶颈
        self.conv1 = nn.Conv2d(num_input_features, 4 * growth_rate, kernel_size=1, bias=False)
        
        # 特征提取：3x3 卷积
        self.bn2 = nn.BatchNorm2d(4 * growth_rate)
        self.relu2 = nn.ReLU(inplace=True)
        # 产生 growth_rate 个新特征图
        self.conv2 = nn.Conv2d(4 * growth_rate, growth_rate, kernel_size=3, padding=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播：提取特征并与输入连接。
        
        参数:
            x: 输入张量，形状为 (B, num_input_features, H, W)
            
        返回:
            连接的张量，形状为 (B, num_input_features + growth_rate, H, W)
        """
        # 瓶颈：减少计算量
        out = self.bn1(x)
        out = self.relu1(out)
        out = self.conv1(out)  # 1x1 卷积：扩展到 4*growth_rate
        
        # 特征提取
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.conv2(out)  # 3x3 卷积：产生 growth_rate 个特征
        
        # 将新特征与所有之前的特征连接
        return torch.cat([x, out], 1)


class Transition(nn.Module):
    """DenseNet 中使用的过渡层。
    
    过渡层放置在密集块之间，用于：
    1. 减少特征图数量（压缩）
    2. 减少空间维度（通过池化下采样）
    
    这有助于控制模型复杂度和内存使用。
    """

    def __init__(self, num_input_features: int, num_output_features: int) -> None:
        """初始化过渡层。
        
        参数:
            num_input_features: 输入特征图数量
            num_output_features: 输出特征图数量（通常为 input // 2）
        """
        super().__init__()
        self.bn = nn.BatchNorm2d(num_input_features)
        self.relu = nn.ReLU(inplace=True)
        # 1x1 卷积压缩特征图
        self.conv = nn.Conv2d(num_input_features, num_output_features, kernel_size=1, bias=False)
        # 平均池化下采样空间维度
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播：压缩和下采样。
        
        参数:
            x: 输入张量，形状为 (B, num_input_features, H, W)
            
        返回:
            输出张量，形状为 (B, num_output_features, H/2, W/2)
        """
        x = self.bn(x)
        x = self.relu(x)
        x = self.conv(x)  # 压缩通道
        x = self.pool(x)  # 下采样空间大小
        return x


class DenseNet121(nn.Module):
    """从零实现的 DenseNet-121。
    
    DenseNet 以前馈方式将每一层连接到所有其他层。
    对于每一层，所有前序层的特征图都用作输入，
    其自己的特征图用作所有后续层的输入。
    
    DenseNet-121 架构：
    - 初始卷积 + 最大池化
    - 密集块 1: 6 个密集层
    - 过渡层 1: 压缩和下采样
    - 密集块 2: 12 个密集层
    - 过渡层 2: 压缩和下采样
    - 密集块 3: 24 个密集层
    - 过渡层 3: 压缩和下采样
    - 密集块 4: 16 个密集层
    - 最终分类器
    
    总计: 6 + 12 + 24 + 16 = 58 个密集层 + 3 个过渡层 = 121 层
    """

    def __init__(self, num_classes: int = 2, growth_rate: int = 32) -> None:
        """初始化 DenseNet-121。
        
        参数:
            num_classes: 输出类别数（默认: 2，用于二分类）
            growth_rate: 每个密集层产生的特征图数量（默认: 32）
        """
        super().__init__()
        self.growth_rate = growth_rate

        # 初始特征提取：大核卷积 + 最大池化
        num_features = 2 * growth_rate  # 从 2*growth_rate 通道开始
        layers: list[nn.Module] = [
            nn.Conv2d(3, num_features, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(num_features),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),  # 下采样
        ]

        # 构建带过渡层的密集块
        # 密集块 1: 6 个密集层
        num_features = self._make_dense_block(layers, num_features, 6)
        num_features = self._make_transition(layers, num_features, num_features // 2)  # 压缩 2 倍
        
        # 密集块 2: 12 个密集层
        num_features = self._make_dense_block(layers, num_features, 12)
        num_features = self._make_transition(layers, num_features, num_features // 2)
        
        # 密集块 3: 24 个密集层
        num_features = self._make_dense_block(layers, num_features, 24)
        num_features = self._make_transition(layers, num_features, num_features // 2)
        
        # 密集块 4: 16 个密集层
        num_features = self._make_dense_block(layers, num_features, 16)

        # 最终批归一化和激活
        layers.append(nn.BatchNorm2d(num_features))
        layers.append(nn.ReLU(inplace=True))

        self.features = nn.Sequential(*layers)

        # 分类器头
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # 全局平均池化
        self.classifier = nn.Linear(num_features, num_classes)

        self._initialize_weights()

    def _make_dense_block(self, layers: list[nn.Module], num_input_features: int, num_layers: int) -> int:
        """创建包含多个密集层的密集块。
        
        参数:
            layers: 要追加层的列表
            num_input_features: 起始特征图数量
            num_layers: 此块中的密集层数量
            
        返回:
            所有层之后的最终特征图数量
            (num_input_features + num_layers * growth_rate)
        """
        for i in range(num_layers):
            # 每一层接收所有之前的特征加上自己的 growth_rate
            layer = DenseLayer(num_input_features + i * self.growth_rate, self.growth_rate)
            layers.append(layer)
        return num_input_features + num_layers * self.growth_rate

    def _make_transition(self, layers: list[nn.Module], num_input_features: int, num_output_features: int) -> int:
        """在密集块之间创建过渡层。
        
        参数:
            layers: 要追加过渡层的列表
            num_input_features: 输入特征图数量
            num_output_features: 输出特征图数量（通常为 input // 2）
            
        返回:
            输出特征图数量
        """
        trans = Transition(num_input_features, num_output_features)
        layers.append(trans)
        return num_output_features

    def _initialize_weights(self) -> None:
        """使用适当的策略初始化模型权重。"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # ReLU 激活的 Kaiming 初始化
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                # BatchNorm: weight=1, bias=0
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                # 线性层: bias=0（权重使用 PyTorch 默认初始化）
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """通过 DenseNet-121 的前向传播。
        
        参数:
            x: 输入张量，形状为 (B, 3, H, W)
            
        返回:
            输出 logits，形状为 (B, num_classes)
        """
        # 通过密集块提取特征
        x = self.features(x)
        # 全局平均池化: (B, C, H, W) -> (B, C, 1, 1)
        x = self.avgpool(x)
        # 展平: (B, C, 1, 1) -> (B, C)
        x = torch.flatten(x, 1)
        # 分类: (B, C) -> (B, num_classes)
        x = self.classifier(x)
        return x


# ---------------------------- Config --------------------------------- #

DEFAULT_EPOCHS: int = 50
DEFAULT_BATCH_SIZE: int = 64
DEFAULT_IMG_SIZE: int = 224
DEFAULT_NUM_WORKERS: int = 4

# Learning rate and weight decay
LR: float = 1e-3
WD: float = 1e-4

# Early stopping
DEFAULT_PATIENCE: int = 10

# Output filenames
BEST_WEIGHTS_NAME: str = "DenseNet121Model.pth"
BEST_CKPT_NAME: str = "best.ckpt"
LATEST_CKPT_NAME: str = "latest.ckpt"

# --------------------------------------------------------------------- #

console = create_console()


def _ensure_rgb(image: Image.Image) -> Image.Image:
    """Convert grayscale frames to RGB."""
    if getattr(image, "mode", "RGB") != "RGB":
        return image.convert("RGB")
    return image


@dataclass(frozen=True)
class EvalResult:
    """Container for evaluation metrics."""

    acc: float
    loss: float
    total: int
    correct: int


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
    """Build train/validation loaders."""
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
    """Compute top-1 accuracy and mean loss."""
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
    """Single-epoch training loop."""
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
    """Entrypoint: data, model, training loop, early stop, save best."""
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

    # Load custom DenseNet121 (implemented from scratch)
    model = DenseNet121(num_classes=num_classes)
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

