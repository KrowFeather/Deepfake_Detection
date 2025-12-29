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


class ConvBNReLU(nn.Sequential):
    """卷积 + 批归一化 + ReLU6 层。
    
    这是 MobileNetV2 中使用的基础构建块，组合了：
    - 可配置步长的 3x3 卷积
    - 批归一化用于稳定训练
    - ReLU6 激活函数（限制在 6）以更好地支持量化
    """

    def __init__(self, in_planes: int, out_planes: int, stride: int = 1) -> None:
        """初始化 ConvBNReLU 块。
        
        参数:
            in_planes: 输入通道数
            out_planes: 输出通道数
            stride: 卷积的步长（默认: 1）
        """
        super().__init__(
            nn.Conv2d(in_planes, out_planes, 3, stride, 1, bias=False),  # 3x3 卷积，padding=1
            nn.BatchNorm2d(out_planes),  # 归一化激活值
            nn.ReLU6(inplace=True),  # ReLU6: max(0, min(x, 6))
        )


class InvertedResidual(nn.Module):
    """MobileNetV2 中使用的倒残差块。
    
    倒残差块遵循以下模式：
    1. 扩展：1x1 逐点卷积扩展通道（如果 expand_ratio > 1）
    2. 深度卷积：3x3 深度可分离卷积
    3. 投影：1x1 逐点卷积减少通道（线性，无激活）
    
    之所以称为"倒"残差，是因为传统残差块先减少后扩展，
    而这个块先扩展后减少。
    """

    def __init__(self, inp: int, oup: int, stride: int, expand_ratio: int) -> None:
        """初始化倒残差块。
        
        参数:
            inp: 输入通道数
            oup: 输出通道数
            stride: 深度卷积的步长
            expand_ratio: 隐藏维度的扩展因子（通常为 6）
        """
        super().__init__()
        self.stride = stride
        assert stride in [1, 2], "步长必须为 1 或 2"

        # 计算隐藏维度：将输入通道数扩展 expand_ratio 倍
        hidden_dim = int(round(inp * expand_ratio))
        # 仅在 stride=1 且输入/输出通道数匹配时使用残差连接
        self.use_res_connect = self.stride == 1 and inp == oup

        layers: list[nn.Module] = []
        # 步骤 1：逐点扩展（1x1 卷积扩展通道）
        if expand_ratio != 1:
            # 扩展通道：inp -> hidden_dim
            layers.append(ConvBNReLU(inp, hidden_dim, stride=1))
        else:
            # 如果不扩展，在这里应用步长
            layers.append(ConvBNReLU(inp, hidden_dim, stride=stride))

        # 步骤 2：深度可分离卷积（3x3 卷积，groups=hidden_dim）
        layers.append(
            nn.Sequential(
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
            )
        )

        # 步骤 3：逐点线性投影（1x1 卷积减少通道，无激活）
        layers.append(
            nn.Sequential(
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),  # 线性投影
                nn.BatchNorm2d(oup),
            )
        )

        self.conv = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """通过倒残差块的前向传播。
        
        参数:
            x: 输入张量，形状为 (B, C, H, W)
            
        返回:
            输出张量，如果适用则包含残差连接
        """
        if self.use_res_connect:
            # 当维度匹配且 stride=1 时添加残差连接
            return x + self.conv(x)
        return self.conv(x)


class MobileNetV2(nn.Module):
    """从零实现的 MobileNetV2。
    
    MobileNetV2 是为移动设备和边缘设备设计的轻量级 CNN 架构。
    主要特点：
    - 使用带深度可分离卷积的倒残差块
    - 宽度乘数允许缩放模型大小（width_mult 参数）
    - 适用于资源受限的环境
    """

    def __init__(self, num_classes: int = 2, width_mult: float = 1.0) -> None:
        """初始化 MobileNetV2。
        
        参数:
            num_classes: 输出类别数（默认: 2，用于二分类）
            width_mult: 宽度乘数，用于缩放模型大小（默认: 1.0 为完整大小）
                        使用 0.5 为一半宽度，0.75 为 75% 宽度等
        """
        super().__init__()
        self.width_mult = width_mult
        # 使通道数能被 8 整除（或非常小的模型为 4）以提高效率
        input_channel = self._make_divisible(32 * width_mult, 4 if width_mult == 0.1 else 8)
        self.last_channel = self._make_divisible(1280 * max(1.0, width_mult), 4 if width_mult == 0.1 else 8)

        # 第一层：标准卷积提取初始特征
        self.features = nn.Sequential(
            ConvBNReLU(3, input_channel, stride=2),  # 下采样 2 倍
        )

        # 倒残差块配置
        # 格式: [扩展比, 输出通道数, 块数量, 步长]
        # t: 扩展比, c: 输出通道数, n: 块数量, s: 步长
        inverted_residual_setting = [
            [1, 16, 1, 1],    # 第一个块：不扩展，16 通道，1 个块，步长 1
            [6, 24, 2, 2],    # 扩展 6 倍，24 通道，2 个块，第一个块步长 2
            [6, 32, 3, 2],    # 扩展 6 倍，32 通道，3 个块，第一个块步长 2
            [6, 64, 4, 2],    # 扩展 6 倍，64 通道，4 个块，第一个块步长 2
            [6, 96, 3, 1],    # 扩展 6 倍，96 通道，3 个块，步长 1（不下采样）
            [6, 160, 3, 2],   # 扩展 6 倍，160 通道，3 个块，第一个块步长 2
            [6, 320, 1, 1],   # 扩展 6 倍，320 通道，1 个块，步长 1
        ]

        # 构建倒残差块
        for t, c, n, s in inverted_residual_setting:
            # 按宽度乘数缩放输出通道数
            output_channel = self._make_divisible(c * width_mult, 4 if width_mult == 0.1 else 8)
            for i in range(n):
                # 只有每组中的第一个块使用指定的步长
                stride = s if i == 0 else 1
                self.features.add_module(
                    f"block{len(self.features)}",
                    InvertedResidual(input_channel, output_channel, stride, expand_ratio=t),
                )
                input_channel = output_channel

        # 分类器之前的最终特征提取层
        self.features.add_module("last_conv", ConvBNReLU(input_channel, self.last_channel, stride=1))

        # 分类器头
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # 全局平均池化
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),  # Dropout 用于正则化
            nn.Linear(self.last_channel, num_classes),
        )

        self._initialize_weights()

    def _make_divisible(self, v: float, divisor: int, min_value: int | None = None) -> int:
        """使值能被除数整除。
        
        这确保通道数能被 8（或 4）整除，以便在受益于对齐内存访问的硬件上高效计算。
        
        参数:
            v: 要使其可整除的值
            divisor: 除数（通常为 8）
            min_value: 允许的最小值（默认: divisor）
            
        返回:
            最接近 v 的可整除值
        """
        if min_value is None:
            min_value = divisor
        # 四舍五入到最接近的可整除值
        new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
        # 确保不会减少太多
        if new_v < 0.9 * v:
            new_v += divisor
        return new_v

    def _initialize_weights(self) -> None:
        """使用适当的初始化策略初始化模型权重。"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # ReLU 激活的 Kaiming 初始化
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                # BatchNorm: weight=1, bias=0
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                # 线性层：小的随机权重
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """通过网络的前向传播。
        
        参数:
            x: 输入张量，形状为 (B, 3, H, W)
            
        返回:
            输出 logits，形状为 (B, num_classes)
        """
        # 通过倒残差块提取特征
        x = self.features(x)
        # 全局平均池化: (B, C, H, W) -> (B, C, 1, 1)
        x = self.avgpool(x)
        # 展平: (B, C, 1, 1) -> (B, C)
        x = torch.flatten(x, 1)
        # 分类: (B, C) -> (B, num_classes)
        x = self.classifier(x)
        return x


def load_mobilenet_v2_pretrained(model: nn.Module, num_classes: int) -> nn.Module:
    """从 torchvision 加载 ImageNet 预训练权重到自定义 MobileNetV2。
    
    此函数将 torchvision 的 MobileNetV2 权重映射到我们的自定义实现。
    主要区别在于命名：
    - torchvision: features.0, features.1, ..., features.18
    - 我们的模型: features.0, features.block1, ..., features.block18, features.last_conv
    
    参数:
        model: 自定义 MobileNetV2 模型实例
        num_classes: 类别数（分类器将被替换，不加载）
    
    返回:
        加载了预训练权重的模型（除了任务特定的分类器）
    """
    # 加载 torchvision 的 ImageNet 预训练 MobileNetV2
    pretrained = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
    pretrained_dict = pretrained.state_dict()
    model_dict = model.state_dict()
    
    # 移除分类器权重（我们将为目标任务使用自己的分类器）
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if 'classifier' not in k}
    
    # 存储成功映射的权重的字典
    mapped_dict = {}
    
    # 映射 features.0（第一个卷积层）- 命名直接匹配
    for k in pretrained_dict.keys():
        if k.startswith('features.0.'):
            if k in model_dict and pretrained_dict[k].shape == model_dict[k].shape:
                mapped_dict[k] = pretrained_dict[k]
    
    # 映射倒残差块
    # torchvision 命名: features.1, features.2, ..., features.18
    # 我们的命名: features.block1, features.block2, ..., features.block18, features.last_conv
    for tv_key, tv_weight in pretrained_dict.items():
        if not tv_key.startswith('features.'):
            continue
        
        parts = tv_key.split('.')
        if len(parts) < 2:
            continue
        
        # 跳过 features.0（上面已处理）
        if parts[1] == '0':
            continue
        
        # 处理 last_conv: torchvision 的 features.18 -> 我们的 features.last_conv
        if parts[1] == '18':
            suffix = '.'.join(parts[2:])  # 获取路径的其余部分
            our_key = f'features.last_conv.{suffix}'
            if our_key in model_dict and tv_weight.shape == model_dict[our_key].shape:
                mapped_dict[our_key] = tv_weight
            continue
        
        # 处理块: features.1-17 -> features.block1-block17
        if parts[1].isdigit():
            block_idx = int(parts[1])
            block_name = f'block{block_idx}'
            suffix = '.'.join(parts[2:])
            our_key = f'features.{block_name}.{suffix}'
            
            if our_key in model_dict:
                if tv_weight.shape == model_dict[our_key].shape:
                    # 形状匹配，可以安全加载
                    mapped_dict[our_key] = tv_weight
                # 注意：深度卷积可能由于 groups 参数而有形状差异
                # 如果形状不匹配，跳过（我们的实现可能略有不同）
    
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
BEST_WEIGHTS_NAME: str = "MobileNetV2Model.pth"
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

    # Load custom MobileNetV2 (implemented from scratch)
    model = MobileNetV2(num_classes=num_classes)
    
    # Load ImageNet pretrained weights if available
    try:
        model = load_mobilenet_v2_pretrained(model, num_classes)
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

