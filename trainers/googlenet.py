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
from torchvision import datasets, models, transforms
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


class BasicConv2d(nn.Module):
    """带批归一化和 ReLU 的基础卷积层。
    
    这是 GoogLeNet 中的标准构建块，组合了：
    - 卷积层（可配置的核大小、步长、填充）
    - 批归一化，epsilon=0.001（GoogLeNet 特定）
    - ReLU 激活
    """

    def __init__(self, in_channels: int, out_channels: int, **kwargs: int) -> None:
        """初始化 BasicConv2d。
        
        参数:
            in_channels: 输入通道数
            out_channels: 输出通道数
            **kwargs: 传递给 Conv2d 的额外参数（kernel_size, stride, padding 等）
        """
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)  # eps=0.001 是 GoogLeNet 特定的
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播: 卷积 -> 批归一化 -> ReLU。"""
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class Inception(nn.Module):
    """Inception 模块（也称为 Inception v1）。
    
    Inception 模块使用多个并行分支，具有不同的滤波器大小，
    以捕获多个尺度的特征：
    - 分支 1: 1x1 卷积（捕获局部特征）
    - 分支 2: 1x1 -> 3x3 卷积（捕获中等尺度特征）
    - 分支 3: 1x1 -> 5x5 卷积（捕获大尺度特征，实现为 3x3）
    - 分支 4: MaxPool -> 1x1 卷积（捕获池化特征）
    
    所有分支沿通道维度连接。
    """

    def __init__(
        self,
        in_channels: int,
        ch1x1: int,
        ch3x3red: int,
        ch3x3: int,
        ch5x5red: int,
        ch5x5: int,
        pool_proj: int,
    ) -> None:
        """初始化 Inception 模块。
        
        参数:
            in_channels: 输入通道数
            ch1x1: 1x1 分支的输出通道数
            ch3x3red: 3x3 卷积前的缩减通道数
            ch3x3: 3x3 分支的输出通道数
            ch5x5red: 5x5 卷积前的缩减通道数（实际为 3x3）
            ch5x5: 5x5 分支的输出通道数（实际为 3x3）
            pool_proj: 池化分支的输出通道数
        """
        super().__init__()
        # 分支 1: 仅 1x1 卷积
        self.branch1 = BasicConv2d(in_channels, ch1x1, kernel_size=1)

        # 分支 2: 1x1 缩减 -> 3x3 卷积
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channels, ch3x3red, kernel_size=1),  # 先缩减通道
            BasicConv2d(ch3x3red, ch3x3, kernel_size=3, padding=1),  # 然后 3x3 卷积
        )

        # 分支 3: 1x1 缩减 -> 5x5 卷积（为效率实现为 3x3）
        self.branch3 = nn.Sequential(
            BasicConv2d(in_channels, ch5x5red, kernel_size=1),  # 先缩减通道
            BasicConv2d(ch5x5red, ch5x5, kernel_size=3, padding=1),  # 5x5 近似为 3x3
        )

        # 分支 4: MaxPool -> 1x1 卷积
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1, ceil_mode=True),
            BasicConv2d(in_channels, pool_proj, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """通过所有分支的前向传播并连接。
        
        参数:
            x: 输入张量，形状为 (B, C, H, W)
            
        返回:
            连接的输出张量，形状为 (B, C_out, H, W)
            其中 C_out = ch1x1 + ch3x3 + ch5x5 + pool_proj
        """
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)

        # 沿通道维度连接所有分支
        outputs = [branch1, branch2, branch3, branch4]
        return torch.cat(outputs, 1)


class InceptionAux(nn.Module):
    """GoogLeNet 的辅助分类器。
    
    辅助分类器在训练时放置在中间层，
    以提供额外的监督信号并帮助梯度流动。
    它们仅在训练时使用，推理时不使用。
    """

    def __init__(self, in_channels: int, num_classes: int) -> None:
        """初始化辅助分类器。
        
        参数:
            in_channels: 输入通道数
            num_classes: 输出类别数
        """
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d((4, 4))  # 池化到 4x4 空间大小
        self.conv = BasicConv2d(in_channels, 128, kernel_size=1)  # 1x1 卷积: 128 通道
        # 4x4x128 = 2048 个特征（展平后）
        self.fc1 = nn.Linear(2048, 1024)
        self.fc2 = nn.Linear(1024, num_classes)
        self.dropout = nn.Dropout(0.7)  # 高 dropout 用于正则化
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """通过辅助分类器的前向传播。
        
        参数:
            x: 输入张量，形状为 (B, C, H, W)
            
        返回:
            类别 logits，形状为 (B, num_classes)
        """
        x = self.avgpool(x)  # (B, C, H, W) -> (B, C, 4, 4)
        x = self.conv(x)  # (B, C, 4, 4) -> (B, 128, 4, 4)
        x = torch.flatten(x, 1)  # (B, 128, 4, 4) -> (B, 2048)
        x = self.fc1(x)  # (B, 2048) -> (B, 1024)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)  # (B, 1024) -> (B, num_classes)
        return x


class GoogLeNet(nn.Module):
    """从零实现的 GoogLeNet (Inception v1)。
    
    GoogLeNet 是 ILSVRC 2014 的获胜者，引入了 Inception 模块，
    使网络能够高效地捕获多尺度特征。
    主要特点：
    - Inception 模块用于多尺度特征提取
    - 辅助分类器用于训练稳定性
    - 与 VGG 相比，参数使用更高效
    """

    def __init__(self, num_classes: int = 2, aux_logits: bool = True) -> None:
        """初始化 GoogLeNet。
        
        参数:
            num_classes: 输出类别数（默认: 2，用于二分类）
            aux_logits: 训练时是否使用辅助分类器（默认: True）
        """
        super().__init__()
        self.aux_logits = aux_logits

        # 初始特征提取层（在 Inception 模块之前）
        self.conv1 = BasicConv2d(3, 64, kernel_size=7, stride=2, padding=3)  # 大核，步长 2
        self.maxpool1 = nn.MaxPool2d(3, stride=2, ceil_mode=True)  # 下采样
        self.conv2 = BasicConv2d(64, 64, kernel_size=1)  # 1x1 卷积用于通道缩减
        self.conv3 = BasicConv2d(64, 192, kernel_size=3, padding=1)  # 扩展通道
        self.maxpool2 = nn.MaxPool2d(3, stride=2, ceil_mode=True)  # 下采样

        # Inception 块组 3
        self.inception3a = Inception(192, 64, 96, 128, 16, 32, 32)
        self.inception3b = Inception(256, 128, 128, 192, 32, 96, 64)
        self.maxpool3 = nn.MaxPool2d(3, stride=2, ceil_mode=True)

        # Inception 块组 4（带辅助分类器）
        self.inception4a = Inception(480, 192, 96, 208, 16, 48, 64)
        self.inception4b = Inception(512, 160, 112, 224, 24, 64, 64)
        self.inception4c = Inception(512, 128, 128, 256, 24, 64, 64)
        self.inception4d = Inception(512, 112, 144, 288, 32, 64, 64)
        self.inception4e = Inception(528, 256, 160, 320, 32, 128, 128)
        self.maxpool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        # Inception 块组 5
        self.inception5a = Inception(832, 256, 160, 320, 32, 128, 128)
        self.inception5b = Inception(832, 384, 192, 384, 48, 128, 128)

        # 辅助分类器（仅在训练时使用）
        if aux_logits:
            self.aux1 = InceptionAux(512, num_classes)  # 在 inception4a 之后
            self.aux2 = InceptionAux(528, num_classes)  # 在 inception4d 之后

        # 主分类器
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # 全局平均池化
        self.dropout = nn.Dropout(0.4)  # Dropout 用于正则化
        self.fc = nn.Linear(1024, num_classes)  # 最终分类层

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """初始化权重。"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, mean=0.0, std=0.01, a=-2, b=2)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """通过 GoogLeNet 的前向传播。
        
        参数:
            x: 输入张量，形状为 (B, 3, H, W)
            
        返回:
            训练时且 aux_logits=True: 元组 (main_output, aux1_output, aux2_output)
            否则: 仅 main_output
            所有输出的形状为 (B, num_classes)
        """
        # 初始特征提取
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.maxpool2(x)

        # Inception 块组 3
        x = self.inception3a(x)
        x = self.inception3b(x)
        x = self.maxpool3(x)

        # Inception 块组 4（带辅助分类器）
        x = self.inception4a(x)
        if self.training and self.aux_logits:
            aux1 = self.aux1(x)  # 第一个辅助分类器

        x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.inception4d(x)
        if self.training and self.aux_logits:
            aux2 = self.aux2(x)  # 第二个辅助分类器

        x = self.inception4e(x)
        x = self.maxpool4(x)

        # Inception 块组 5
        x = self.inception5a(x)
        x = self.inception5b(x)

        # 主分类器
        x = self.avgpool(x)  # 全局平均池化
        x = torch.flatten(x, 1)  # 展平为 (B, 1024)
        x = self.dropout(x)
        x = self.fc(x)  # 最终分类

        # 训练时返回辅助输出以提供额外监督
        if self.training and self.aux_logits:
            return x, aux1, aux2
        return x


def load_googlenet_pretrained(model: nn.Module, num_classes: int) -> nn.Module:
    """从 torchvision 加载 ImageNet 预训练权重到自定义 GoogLeNet。
    
    此函数将 torchvision 的 GoogLeNet 权重映射到我们的自定义实现。
    大多数层名称直接匹配，但我们跳过：
    - 辅助分类器 (aux1, aux2) - 任务特定
    - 最终分类器 (fc) - 任务特定
    
    参数:
        model: 自定义 GoogLeNet 模型实例
        num_classes: 类别数（分类器将被替换，不加载）
    
    返回:
        加载了预训练权重的模型（除了分类器和辅助分类器）
    """
    # 加载 torchvision 的 ImageNet 预训练 GoogLeNet
    pretrained = models.googlenet(weights=models.GoogLeNet_Weights.IMAGENET1K_V1)
    pretrained_dict = pretrained.state_dict()
    model_dict = model.state_dict()
    
    # 创建从 torchvision 键到我们键的映射
    # 大多数键直接匹配（例如 'conv1.conv', 'conv1.bn', 'inception3a.branch1.conv' 等）
    key_mapping = {}
    
    # 映射两个模型中都存在的所有键
    for pretrained_key in pretrained_dict.keys():
        # 跳过辅助分类器 (aux1, aux2) - 这些是任务特定的
        if 'aux' in pretrained_key:
            continue
        # 跳过最终分类器 - 我们将为目标任务使用自己的分类器
        if pretrained_key in ['fc.weight', 'fc.bias']:
            continue
        
        # 如果键在我们的模型中存在，添加到映射
        if pretrained_key in model_dict:
            key_mapping[pretrained_key] = pretrained_key
    
    # 加载形状和名称都匹配的权重
    mapped_dict = {}
    for pretrained_key, model_key in key_mapping.items():
        if pretrained_dict[pretrained_key].shape == model_dict[model_key].shape:
            # 形状匹配，可以安全加载
            mapped_dict[model_key] = pretrained_dict[pretrained_key]
    
    # 用成功映射的权重更新模型
    model_dict.update(mapped_dict)
    model.load_state_dict(model_dict, strict=False)  # strict=False 允许缺失/不匹配的键
    
    return model


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
BEST_WEIGHTS_NAME: str = "GoogLeNetModel.pth"
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
            # GoogLeNet returns main output only during eval
            logits = model(inputs)
            if isinstance(logits, tuple):
                logits = logits[0]  # Use main output
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
            outputs = model(inputs)
            # GoogLeNet returns (main, aux1, aux2) during training
            if isinstance(outputs, tuple):
                main_output, aux1_output, aux2_output = outputs
                loss = criterion(main_output, targets) + 0.3 * criterion(aux1_output, targets) + 0.3 * criterion(aux2_output, targets)
            else:
                loss = criterion(outputs, targets)

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

    # Load custom GoogLeNet (implemented from scratch)
    model = GoogLeNet(num_classes=num_classes, aux_logits=True)
    
    # Load ImageNet pretrained weights if available
    try:
        model = load_googlenet_pretrained(model, num_classes)
        console.print("[bold green]✓ Loaded ImageNet pretrained weights[/]")
    except Exception as e:
        console.print(f"[bold yellow]⚠ Could not load pretrained weights:[/] {e}")
        console.print("[yellow]Training from scratch...[/]")
    
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

