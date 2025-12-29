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


class Swish(nn.Module):
    """Swish 激活函数: x * sigmoid(x)。"""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)


class SEBlock(nn.Module):
    """Squeeze-and-Excitation 注意力模块。
    
    通过全局平均池化和两个全连接层来重新校准通道特征。
    """

    def __init__(self, in_channels: int, se_ratio: float = 0.25) -> None:
        """初始化 SE 块。
        
        参数:
            in_channels: 输入通道数
            se_ratio: 压缩比，用于确定中间层通道数
        """
        super().__init__()
        reduced_channels = max(1, int(in_channels * se_ratio))
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # 全局平均池化
            nn.Conv2d(in_channels, reduced_channels, 1, bias=False),  # 压缩
            Swish(),  # 激活
            nn.Conv2d(reduced_channels, in_channels, 1, bias=False),  # 扩展
            nn.Sigmoid(),  # 归一化到 [0, 1]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播：应用通道注意力。"""
        return x * self.se(x)


class MBConvBlock(nn.Module):
    """Mobile Inverted Residual Bottleneck 块（EfficientNet 的核心构建块）。
    
    包含：
    1. 可选的扩展卷积（1x1，如果 expand_ratio > 1）
    2. 深度卷积（3x3 或 5x5）
    3. SE 注意力模块（可选）
    4. 投影卷积（1x1，线性）
    5. 残差连接（如果 stride=1 且输入输出通道相同）
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        expand_ratio: int,
        se_ratio: float = 0.25,
        drop_connect_rate: float = 0.0,
    ) -> None:
        """初始化 MBConv 块。
        
        参数:
            in_channels: 输入通道数
            out_channels: 输出通道数
            kernel_size: 深度卷积的核大小（3 或 5）
            stride: 深度卷积的步长
            expand_ratio: 扩展比（隐藏层通道数 = in_channels * expand_ratio）
            se_ratio: SE 模块的压缩比
            drop_connect_rate: DropConnect 的丢弃率（用于训练）
        """
        super().__init__()
        self.stride = stride
        self.expand_ratio = expand_ratio
        self.drop_connect_rate = drop_connect_rate

        # 计算隐藏层通道数
        hidden_channels = in_channels * expand_ratio
        self.use_residual = stride == 1 and in_channels == out_channels

        layers: list[nn.Module] = []

        # 扩展阶段（如果 expand_ratio > 1）
        if expand_ratio != 1:
            layers.extend([
                nn.Conv2d(in_channels, hidden_channels, 1, bias=False),
                nn.BatchNorm2d(hidden_channels),
                Swish(),
            ])

        # 深度卷积阶段
        padding = (kernel_size - 1) // 2
        layers.extend([
            nn.Conv2d(
                hidden_channels if expand_ratio != 1 else in_channels,
                hidden_channels,
                kernel_size,
                stride,
                padding,
                groups=hidden_channels if expand_ratio != 1 else in_channels,
                bias=False,
            ),
            nn.BatchNorm2d(hidden_channels),
            Swish(),
        ])

        # SE 注意力模块
        if se_ratio > 0:
            layers.append(SEBlock(hidden_channels, se_ratio))

        # 投影阶段（线性，无激活）
        layers.extend([
            nn.Conv2d(hidden_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
        ])

        self.conv = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播：通过 MBConv 块。"""
        out = self.conv(x)

        # DropConnect（训练时随机丢弃连接）
        if self.use_residual and self.training and self.drop_connect_rate > 0:
            out = self._drop_connect(out)

        # 残差连接
        if self.use_residual:
            out = out + x

        return out

    def _drop_connect(self, x: torch.Tensor) -> torch.Tensor:
        """DropConnect：随机丢弃连接。"""
        if not self.training:
            return x
        random_tensor = torch.rand_like(x)
        binary_tensor = random_tensor > self.drop_connect_rate
        return x * binary_tensor / (1.0 - self.drop_connect_rate)


class EfficientNet(nn.Module):
    """EfficientNet 从零实现。
    
    EfficientNet 通过复合缩放（深度、宽度、分辨率）来平衡模型效率和性能。
    这里实现 EfficientNet-B2 的基础架构。
    """

    def __init__(
        self,
        num_classes: int = 2,
        width_mult: float = 1.1,  # B2 的默认宽度乘数
        depth_mult: float = 1.2,  # B2 的默认深度乘数
        dropout_rate: float = 0.3,  # B2 的默认 dropout
        drop_connect_rate: float = 0.2,
    ) -> None:
        """初始化 EfficientNet。
        
        参数:
            num_classes: 输出类别数（默认: 2，用于二分类）
            width_mult: 宽度乘数（缩放通道数，B2 默认 1.1）
            depth_mult: 深度乘数（缩放层数，B2 默认 1.2）
            dropout_rate: 最终分类器的 dropout 率（B2 默认 0.3）
            drop_connect_rate: MBConv 块的 DropConnect 率
        """
        super().__init__()

        # EfficientNet-B2 的基础配置
        # [expand_ratio, kernel_size, channels, num_layers, stride]
        # B2 的配置与 B0 相同，但通过 width_mult 和 depth_mult 进行缩放
        base_config = [
            [1, 3, 32, 1, 2],   # 初始 MBConv
            [6, 3, 16, 1, 1],   # Stage 1
            [6, 3, 24, 2, 2],   # Stage 2
            [6, 5, 40, 2, 2],   # Stage 3
            [6, 3, 80, 3, 2],   # Stage 4
            [6, 5, 112, 3, 1],  # Stage 5
            [6, 5, 192, 4, 2],  # Stage 6
            [6, 3, 320, 1, 1],  # Stage 7
        ]

        # 初始卷积层
        in_channels = self._round_channels(32 * width_mult)
        self.conv_stem = nn.Sequential(
            nn.Conv2d(3, in_channels, 3, 2, 1, bias=False),
            nn.BatchNorm2d(in_channels),
            Swish(),
        )

        # 构建 MBConv 块
        blocks: list[nn.Module] = []
        num_blocks = sum(config[3] for config in base_config)
        block_idx = 0

        for expand_ratio, kernel_size, channels, num_layers, stride in base_config:
            out_channels = self._round_channels(channels * width_mult)
            num_layers = self._round_repeats(num_layers * depth_mult)

            # 第一个块使用指定的 stride
            blocks.append(
                MBConvBlock(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride,
                    expand_ratio,
                    drop_connect_rate=drop_connect_rate * block_idx / num_blocks,
                )
            )
            block_idx += 1
            in_channels = out_channels

            # 剩余块使用 stride=1
            for _ in range(num_layers - 1):
                blocks.append(
                    MBConvBlock(
                        in_channels,
                        out_channels,
                        kernel_size,
                        1,
                        expand_ratio,
                        drop_connect_rate=drop_connect_rate * block_idx / num_blocks,
                    )
                )
                block_idx += 1

        self.blocks = nn.Sequential(*blocks)

        # 最终特征提取
        final_channels = self._round_channels(1280 * width_mult)
        self.conv_head = nn.Sequential(
            nn.Conv2d(in_channels, final_channels, 1, bias=False),
            nn.BatchNorm2d(final_channels),
            Swish(),
        )

        # 分类器
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(final_channels, num_classes)

        self._initialize_weights()

    def _round_channels(self, channels: float) -> int:
        """将通道数四舍五入到 8 的倍数（硬件优化）。"""
        return int(round(channels + 0.5))

    def _round_repeats(self, repeats: float) -> int:
        """将重复次数向上取整。"""
        return int(round(repeats + 0.5))

    def _initialize_weights(self) -> None:
        """初始化模型权重。"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播。
        
        参数:
            x: 输入张量，形状为 (B, 3, H, W)
            
        返回:
            输出 logits，形状为 (B, num_classes)
        """
        # 初始特征提取
        x = self.conv_stem(x)

        # MBConv 块
        x = self.blocks(x)

        # 最终特征提取
        x = self.conv_head(x)

        # 全局平均池化和分类
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.classifier(x)

        return x


def load_efficientnet_pretrained(model: nn.Module, num_classes: int) -> nn.Module:
    """从 torchvision 加载 ImageNet 预训练权重到自定义 EfficientNet。
    
    此函数将 torchvision 的 EfficientNet-B2 权重映射到我们的自定义实现。
    主要映射关系：
    - features.0 -> conv_stem
    - features.1-N -> blocks
    - features.N+1 -> conv_head
    - classifier -> 跳过（使用自定义分类器）
    
    参数:
        model: 自定义 EfficientNet 模型实例
        num_classes: 类别数（分类器将被替换，不加载）
    
    返回:
        加载了预训练权重的模型（除了任务特定的分类器）
    """
    try:
        # 加载 torchvision 的预训练 EfficientNet-B2
        pretrained = models.efficientnet_b2(weights=models.EfficientNet_B2_Weights.IMAGENET1K_V1)
        pretrained_dict = pretrained.state_dict()
        model_dict = model.state_dict()

        # 移除分类器权重
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if "classifier" not in k}

        # 创建权重映射字典
        mapped_dict = {}

        # 映射初始卷积层 (features.0 -> conv_stem)
        for k in list(pretrained_dict.keys()):
            if k.startswith("features.0."):
                new_key = k.replace("features.0", "conv_stem")
                if new_key in model_dict and pretrained_dict[k].shape == model_dict[new_key].shape:
                    mapped_dict[new_key] = pretrained_dict[k]

        # 映射 MBConv 块 (features.1-N -> blocks)
        # torchvision 的 features 索引从 1 开始，对应我们的 blocks
        block_idx = 0
        for i in range(1, len(pretrained.features)):  # type: ignore[attr-defined]
            if i == len(pretrained.features) - 1:  # type: ignore[attr-defined]
                # 最后一个可能是 conv_head
                continue

            # 查找对应的块
            for k in list(pretrained_dict.keys()):
                if k.startswith(f"features.{i}."):
                    # 尝试映射到 blocks
                    suffix = k.split(f"features.{i}.")[1]
                    new_key = f"blocks.{block_idx}.{suffix}"
                    if new_key in model_dict and pretrained_dict[k].shape == model_dict[new_key].shape:
                        mapped_dict[new_key] = pretrained_dict[k]

            # 检查是否是一个完整的块（通过检查是否有 conv 层）
            has_conv = any(k.startswith(f"features.{i}.block") for k in pretrained_dict.keys())
            if has_conv:
                block_idx += 1

        # 映射最终卷积层 (features.N -> conv_head)
        last_idx = len(pretrained.features) - 1  # type: ignore[attr-defined]
        for k in list(pretrained_dict.keys()):
            if k.startswith(f"features.{last_idx}."):
                new_key = k.replace(f"features.{last_idx}", "conv_head")
                if new_key in model_dict and pretrained_dict[k].shape == model_dict[new_key].shape:
                    mapped_dict[new_key] = pretrained_dict[k]

        # 更新模型权重
        model_dict.update(mapped_dict)
        model.load_state_dict(model_dict, strict=False)

        print(f"[EfficientNet] 成功加载 {len(mapped_dict)}/{len(pretrained_dict)} 个预训练权重层")

    except Exception as e:
        print(f"[EfficientNet] 预训练权重加载失败: {e}，使用随机初始化")

    return model


# ---------------------------- Config --------------------------------- #

DATA_ROOT: Path = Path.home() / "code" / "DeepfakeDetection" / "data" / "Dataset"

DEFAULT_EPOCHS: int = 25
DEFAULT_BATCH_SIZE: int = 64
DEFAULT_IMG_SIZE: int = 224
DEFAULT_NUM_WORKERS: int = 8

HEAD_LR: float = 3e-4
HEAD_WD: float = 5e-2
FT_LR: float = 1e-4
FT_WD: float = 5e-2

DEFAULT_PATIENCE: int = 4

BEST_WEIGHTS_NAME: str = "EfficientNetFromScratch.pth"
BEST_CKPT_NAME: str = "best.ckpt"
LATEST_CKPT_NAME: str = "latest.ckpt"

FT_BATCH_SIZE: int = 32
EFFECTIVE_BATCH: int = 128
DEFAULT_ACCUM_STEPS: int = max(1, EFFECTIVE_BATCH // FT_BATCH_SIZE)

# --------------------------------------------------------------------- #

console = create_console()


def _ensure_rgb(image: Image.Image) -> Image.Image:
    """将灰度图像转换为 RGB。"""
    if getattr(image, "mode", "RGB") != "RGB":
        return image.convert("RGB")  # type: ignore[no-any-return]
    return image


@dataclass(frozen=True)
class EvalResult:
    """评估结果容器。"""

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
    small_images = img_size <= 64

    defaults = {
        "ensure_rgb": True,
        "train_resize": True,
        "train_random_crop": small_images,
        "train_center_crop": False,
        "train_random_resized_crop": not small_images,
        "train_random_horizontal_flip": True,
        "train_random_rotation": not small_images,
        "train_color_jitter": not small_images,
        "train_random_erasing": not small_images,
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

    if small_images:
        if toggles.get("train_resize", True):
            train_ops.append(
                transforms.Resize(img_size + 4, interpolation=InterpolationMode.BILINEAR),
            )
        if toggles.get("train_random_crop", True):
            train_ops.append(transforms.RandomCrop(img_size))
        elif toggles.get("train_center_crop", False):
            train_ops.append(transforms.CenterCrop(img_size))
    else:
        resize_shorter = max(img_size + 32, int(img_size * 1.15))
        if toggles.get("train_random_resized_crop", True):
            train_ops.append(transforms.RandomResizedCrop(img_size, scale=(0.9, 1.0)))
        else:
            if toggles.get("train_resize", True):
                train_ops.append(
                    transforms.Resize(resize_shorter, interpolation=InterpolationMode.BILINEAR),
                )
            if toggles.get("train_center_crop", True):
                train_ops.append(transforms.CenterCrop(img_size))
        if toggles.get("train_random_rotation", True):
            train_ops.append(transforms.RandomRotation(10))

    if toggles.get("train_random_horizontal_flip", True):
        train_ops.append(transforms.RandomHorizontalFlip())
    if toggles.get("train_color_jitter", False):
        train_ops.append(transforms.ColorJitter(0.2, 0.2, 0.2, 0.05))
    if toggles.get("train_to_tensor", True):
        train_ops.append(transforms.ToTensor())
    if toggles.get("train_normalize", True):
        train_ops.append(normalize)
    if toggles.get("train_random_erasing", False):
        train_ops.append(
            transforms.RandomErasing(
                p=0.5,
                scale=(0.02, 0.33),
                ratio=(0.3, 3.3),
                value=0,
            ),
        )

    val_ops: list[object] = []
    if toggles.get("ensure_rgb", True):
        val_ops.append(transforms.Lambda(_ensure_rgb))
    if toggles.get("val_resize", True):
        resize_target = max(img_size + 32, int(img_size * 1.15)) if not small_images else img_size
        val_ops.append(transforms.Resize(resize_target, interpolation=InterpolationMode.BILINEAR))
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
            inputs = batch_x.to(device, non_blocking=True).to(
                memory_format=torch.channels_last,
            )
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


def train_one_epoch(  # noqa: PLR0913
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
    accum_steps: int = 1,
) -> float:
    """单轮训练循环，包含 AMP、梯度累积和实时吞吐量报告。"""
    model.train()
    start = perf_counter()
    opt.zero_grad(set_to_none=True)

    loss_sum = 0.0
    seen_total = 0
    pending_steps = 0

    for i, (batch_x, batch_y) in enumerate(dl, 1):
        inputs = batch_x.to(device, non_blocking=True).to(
            memory_format=torch.channels_last,
        )
        targets = batch_y.to(device, non_blocking=True)

        with torch.amp.autocast(device_type="cuda", enabled=use_cuda_amp):
            logits = model(inputs)
            loss = criterion(logits, targets)
            if accum_steps > 1:
                loss = loss / accum_steps

        scaler.scale(loss).backward()
        pending_steps += 1

        if pending_steps == accum_steps:
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
            pending_steps = 0

        bsz = targets.size(0)
        seen_total += bsz
        loss_sum += float(loss.item()) * bsz * (max(1, accum_steps))

        elapsed = perf_counter() - start
        seen = min(i * dl.batch_size, len(dl.dataset))
        ips = seen / max(1e-6, elapsed)
        shown_loss = float(loss.item() * (max(1, accum_steps)))
        progress.update(
            task,
            advance=1,
            description=f"train | loss={shown_loss:.4f} | {ips:.0f} img/s",
        )

    if pending_steps > 0:
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)

    return loss_sum / max(1, seen_total)


def main() -> None:  # noqa: PLR0915
    """入口点：数据、模型、预热、微调、早停、保存最佳。"""
    env = prepare_training_environment(
        weights_name=BEST_WEIGHTS_NAME,
        best_checkpoint_name=BEST_CKPT_NAME,
        latest_checkpoint_name=LATEST_CKPT_NAME,
    )
    apply_seed(env.seed)

    data_root = env_path("DATA_ROOT", DATA_ROOT)
    train_split = env_str("TRAIN_SPLIT", "Train")
    val_split = env_str("VAL_SPLIT", "Validation")
    batch_size = env_int("BATCH_SIZE", DEFAULT_BATCH_SIZE)
    epochs = env_int("EPOCHS", DEFAULT_EPOCHS)
    img_size = env_int("IMG_SIZE", DEFAULT_IMG_SIZE)
    num_workers = env_int("NUM_WORKERS", DEFAULT_NUM_WORKERS)
    num_classes = env_int("NUM_CLASSES", 2)
    accum_steps = env_int("ACCUM_STEPS", DEFAULT_ACCUM_STEPS)
    ft_lr = env_float("LR", FT_LR)
    ft_wd = env_float("WEIGHT_DECAY", FT_WD)
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
        console.print(
            "Update `data.num_classes` in your YAML to match the dataset. "
            "For MNIST, set it to 10.",
        )
        raise SystemExit(1) from exc
    console.print(
        f"[bold]Data[/]: train={len(train_dl.dataset)} | val={len(val_dl.dataset)} | "
        f"bs={batch_size} | steps/epoch={len(train_dl)}",
    )

    model = EfficientNet(num_classes=num_classes)
    # 尝试加载预训练权重（如果可用）
    model = load_efficientnet_pretrained(model, num_classes)

    model.to(memory_format=torch.channels_last)
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
    warmup_done = env.resume_checkpoint is not None

    with progress:
        if not warmup_done:
            for p in model.parameters():
                p.requires_grad = False
            for name, param in model.named_parameters():
                if "classifier" in name:
                    param.requires_grad = True

            head_params = [p for p in model.parameters() if p.requires_grad]
            warm_opt = optim.AdamW(head_params, lr=HEAD_LR, weight_decay=HEAD_WD)

            warm_task = progress.add_task(
                "warmup (head only)",
                total=len(train_dl),
                extra="",
            )
            console.print("[bold]Warmup (head only)[/]")
            _ = train_one_epoch(
                model=model,
                dl=train_dl,
                opt=warm_opt,
                scaler=scaler,
                criterion=criterion,
                device=device,
                use_cuda_amp=use_cuda,
                progress=progress,
                task=warm_task,
                accum_steps=1,
            )

            res = evaluate(model, val_dl, device, criterion)
            console.print(
                f"[bold cyan]warmup[/] | val_acc={res.acc:.4f} | val_loss={res.loss:.4f} "
                f"({res.correct}/{res.total})",
            )
            best_val_acc = res.acc
            best_epoch = 0
            warmup_done = True

        for p in model.parameters():
            p.requires_grad = True

        console.print(
            f"[bold]Fine-tune[/]: bs={FT_BATCH_SIZE}, accum_steps={accum_steps} "
            f"(effective ≈ {FT_BATCH_SIZE * accum_steps})",
        )
        train_dl_ft = DataLoader(
            train_dl.dataset,
            batch_size=FT_BATCH_SIZE,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=num_workers > 0,
            prefetch_factor=2,
        )

        opt = optim.AdamW(
            (p for p in model.parameters() if p.requires_grad),
            lr=ft_lr,
            weight_decay=ft_wd,
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs - 1))

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
            warmup_done = bool(resume_state.get("warmup_done", warmup_done))
            epochs_no_improve = max(0, start_epoch - best_epoch)
            console.print(
                f"[bold green]Resumed[/] from epoch {start_epoch} using {env.resume_checkpoint}",
            )

        for epoch in range(start_epoch + 1, epochs + 1):
            task = progress.add_task(f"epoch {epoch}", total=len(train_dl_ft), extra="")
            train_loss = train_one_epoch(
                model=model,
                dl=train_dl_ft,
                opt=opt,
                scaler=scaler,
                criterion=criterion,
                device=device,
                use_cuda_amp=use_cuda,
                progress=progress,
                task=task,
                accum_steps=accum_steps,
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
                extra={"warmup_done": warmup_done},
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

