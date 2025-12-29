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


class ConvBNReLU(nn.Sequential):
    """卷积 + 批归一化 + ReLU 层。"""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1, padding: int = 1) -> None:
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )


class DepthwiseConv2d(nn.Module):
    """深度可分离卷积。"""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1, padding: int = 1) -> None:
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size,
            stride,
            padding,
            groups=in_channels,
            bias=False,
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.act(x)
        return x


class Attention(nn.Module):
    """多头自注意力机制（简化版，用于 EfficientFormer）。"""

    def __init__(self, dim: int, num_heads: int = 8, qkv_bias: bool = False) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x


class MLP(nn.Module):
    """多层感知机（前馈网络）。"""

    def __init__(self, in_features: int, hidden_features: int | None = None, out_features: int | None = None, drop: float = 0.0) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class TransformerBlock(nn.Module):
    """Transformer 块（用于 EfficientFormer 的后半部分）。"""

    def __init__(self, dim: int, num_heads: int = 8, mlp_ratio: float = 4.0, drop: float = 0.0) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(dim, mlp_hidden_dim, drop=drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class EfficientFormerBlock(nn.Module):
    """EfficientFormer 混合块（CNN + Transformer）。
    
    EfficientFormer 使用混合架构：
    - 前半部分使用 CNN（MobileNet 风格）
    - 后半部分使用 Transformer
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        expand_ratio: int = 4,
        use_transformer: bool = False,
        dim: int | None = None,
        num_heads: int = 8,
    ) -> None:
        super().__init__()
        self.use_transformer = use_transformer

        if use_transformer:
            # Transformer 模式：需要将特征图转换为序列
            assert dim is not None, "Transformer 模式需要指定 dim"
            # patch_embed 将输入通道转换为 dim，并可能下采样
            self.patch_embed = nn.Conv2d(in_channels, dim, kernel_size=3, stride=stride, padding=1)
            self.transformer = TransformerBlock(dim, num_heads)
            # proj 将 dim 转换为 out_channels
            self.proj = nn.Linear(dim, out_channels)
            self.out_channels = out_channels
        else:
            # CNN 模式：使用倒残差块
            hidden_dim = int(in_channels * expand_ratio)
            layers: list[nn.Module] = []

            if expand_ratio != 1:
                layers.append(ConvBNReLU(in_channels, hidden_dim, 1, 1, 0))

            layers.append(DepthwiseConv2d(hidden_dim if expand_ratio != 1 else in_channels, hidden_dim, 3, stride, 1))
            layers.append(nn.Conv2d(hidden_dim, out_channels, 1, bias=False))
            layers.append(nn.BatchNorm2d(out_channels))

            self.conv = nn.Sequential(*layers)
            self.use_residual = stride == 1 and in_channels == out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_transformer:
            # Transformer 路径
            B, C, H, W = x.shape
            # patch_embed: (B, in_channels, H, W) -> (B, dim, H', W')
            # 如果 stride=2，H'=H//2, W'=W//2
            x = self.patch_embed(x)
            B_new, C_new, H_new, W_new = x.shape
            # 转换为序列: (B, dim, H', W') -> (B, H'*W', dim)
            x = x.flatten(2).transpose(1, 2)
            # Transformer: (B, H'*W', dim) -> (B, H'*W', dim)
            x = self.transformer(x)
            # 投影: (B, H'*W', dim) -> (B, H'*W', out_channels)
            x = self.proj(x)
            # 重新整形回特征图: (B, H'*W', out_channels) -> (B, out_channels, H', W')
            x = x.transpose(1, 2).reshape(B_new, self.out_channels, H_new, W_new)
            return x
        else:
            # CNN 路径
            out = self.conv(x)
            if self.use_residual:
                out = out + x
            return out


class EfficientFormerV2(nn.Module):
    """EfficientFormerV2 从零实现。
    
    EfficientFormerV2 是一个混合架构，结合了 CNN 和 Transformer 的优势：
    - 前半部分使用轻量级 CNN（类似 MobileNet）
    - 后半部分使用 Transformer 块
    """

    def __init__(self, num_classes: int = 2, embed_dims: list[int] | None = None) -> None:
        """初始化 EfficientFormerV2。
        
        参数:
            num_classes: 输出类别数（默认: 2，用于二分类）
            embed_dims: Transformer 阶段的嵌入维度列表
        """
        super().__init__()

        if embed_dims is None:
            embed_dims = [48, 96, 192, 384]  # EfficientFormerV2-S1 的配置

        # 初始特征提取
        self.stem = nn.Sequential(
            ConvBNReLU(3, 24, 3, 2, 1),  # 下采样到 112x112
            ConvBNReLU(24, 48, 3, 2, 1),  # 下采样到 56x56
        )

        # Stage 1-2: CNN 块
        self.stage1 = nn.Sequential(
            EfficientFormerBlock(48, 48, 1, 4),
            EfficientFormerBlock(48, 48, 1, 4),
        )

        self.stage2 = nn.Sequential(
            EfficientFormerBlock(48, 96, 2, 4),  # 下采样到 28x28，输出 96 通道
            EfficientFormerBlock(96, 96, 1, 4),  # 保持 96 通道
        )

        # Stage 3-4: 混合块（开始使用 Transformer）
        self.stage3 = nn.Sequential(
            EfficientFormerBlock(96, embed_dims[0], 2, 4, use_transformer=True, dim=embed_dims[0]),  # 下采样到 14x14
            EfficientFormerBlock(embed_dims[0], embed_dims[0], 1, 4, use_transformer=True, dim=embed_dims[0]),
        )

        self.stage4 = nn.Sequential(
            EfficientFormerBlock(embed_dims[0], embed_dims[1], 2, 4, use_transformer=True, dim=embed_dims[1]),  # 下采样到 7x7
            EfficientFormerBlock(embed_dims[1], embed_dims[1], 1, 4, use_transformer=True, dim=embed_dims[1]),
        )

        # 分类头
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(embed_dims[1], num_classes)

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """初始化模型权重。"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播。
        
        参数:
            x: 输入张量，形状为 (B, 3, H, W)
            
        返回:
            输出 logits，形状为 (B, num_classes)
        """
        # 初始特征提取
        x = self.stem(x)

        # CNN 阶段
        x = self.stage1(x)
        x = self.stage2(x)

        # Transformer 阶段
        x = self.stage3(x)
        x = self.stage4(x)

        # 分类
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.head(x)

        return x


def load_efficientformer_v2_pretrained(model: nn.Module, num_classes: int) -> nn.Module:
    """从 timm 加载 ImageNet 预训练权重到自定义 EfficientFormerV2。
    
    此函数尝试从 timm 加载 EfficientFormerV2-S1 的预训练权重。
    由于架构可能不完全匹配，会尝试加载兼容的权重。
    
    注意：由于架构差异较大，目前暂时禁用预训练权重加载，使用随机初始化。
    后续可以根据实际架构调整权重映射逻辑。
    
    参数:
        model: 自定义 EfficientFormerV2 模型实例
        num_classes: 类别数（分类器将被替换，不加载）
    
    返回:
        模型（目前返回未加载预训练权重的模型）
    """
    # 暂时禁用预训练权重加载，因为架构差异较大
    # 如果需要加载预训练权重，需要仔细匹配层名称和形状
    print("[EfficientFormerV2] 跳过预训练权重加载（架构差异较大），使用随机初始化")
    
    # 如果需要启用预训练权重加载，可以取消下面的注释并调整映射逻辑
    """
    try:
        import timm

        # 尝试从 timm 加载预训练模型
        pretrained = timm.create_model("efficientformerv2_s1", pretrained=True, num_classes=1000)
        pretrained_dict = pretrained.state_dict()
        model_dict = model.state_dict()

        # 移除分类器权重
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if "head" not in k and "classifier" not in k}

        # 创建权重映射字典
        mapped_dict = {}

        # 尝试映射权重（基于名称相似性）
        # 注意：由于架构可能不完全匹配，我们只加载形状完全匹配的权重
        for pretrained_key, pretrained_weight in pretrained_dict.items():
            # 尝试直接匹配
            if pretrained_key in model_dict:
                if pretrained_weight.shape == model_dict[pretrained_key].shape:
                    mapped_dict[pretrained_key] = pretrained_weight
                # 如果形状不匹配，跳过（不强制加载）
                continue

            # 尝试部分匹配（处理命名差异）
            # 但只匹配形状完全相同的层，避免通道数不匹配
            for model_key in model_dict.keys():
                # 检查是否是同一类型的层（通过后缀匹配）
                pretrained_parts = pretrained_key.split(".")
                model_parts = model_key.split(".")
                
                # 如果最后两层名称匹配
                if len(pretrained_parts) >= 2 and len(model_parts) >= 2:
                    if pretrained_parts[-2:] == model_parts[-2:]:
                        # 只加载形状完全匹配的权重
                        if pretrained_weight.shape == model_dict[model_key].shape:
                            mapped_dict[model_key] = pretrained_weight
                            break

        # 更新模型权重
        model_dict.update(mapped_dict)
        model.load_state_dict(model_dict, strict=False)

        print(f"[EfficientFormerV2] 成功加载 {len(mapped_dict)}/{len(pretrained_dict)} 个预训练权重层")

    except Exception as e:
        print(f"[EfficientFormerV2] 预训练权重加载失败: {e}，使用随机初始化")
    """

    return model


# ---------------------------- Config --------------------------------- #

DATA_ROOT: Path = Path.home() / "code" / "DeepfakeDetection" / "data" / "Dataset"

DEFAULT_EPOCHS: int = 5
DEFAULT_BATCH_SIZE: int = 128
DEFAULT_IMG_SIZE: int = 224
DEFAULT_NUM_WORKERS: int = 8
DEFAULT_LR: float = 1e-4
DEFAULT_WEIGHT_DECAY: float = 5e-2

BEST_WEIGHTS_NAME: str = "EfficientFormerV2_FromScratch.pth"
BEST_CKPT_NAME: str = "best.ckpt"
LATEST_CKPT_NAME: str = "latest.ckpt"

UNFREEZE_KEYS: tuple[str, ...] = (
    "stages.3",
    "blocks.3",
    "stage3",
    "stage4",
    "head",
)

# --------------------------------------------------------------------- #

console = create_console()


def _ensure_rgb(image: Image.Image) -> Image.Image:
    """将灰度图像转换为 RGB。"""
    if getattr(image, "mode", "RGB") != "RGB":
        return image.convert("RGB")  # type: ignore[no-any-return]
    return image


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
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    small_images = img_size <= 64

    defaults = {
        "ensure_rgb": True,
        "train_resize": True,
        "train_random_crop": small_images,
        "train_center_crop": False,
        "train_random_resized_crop": not small_images,
        "train_random_horizontal_flip": True,
        "train_random_rotation": False,
        "train_color_jitter": not small_images,
        "train_random_erasing": False,
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

    if toggles.get("train_random_horizontal_flip", True):
        train_ops.append(transforms.RandomHorizontalFlip())
    if toggles.get("train_random_rotation", False):
        train_ops.append(transforms.RandomRotation(10))
    if toggles.get("train_color_jitter", False):
        train_ops.append(transforms.ColorJitter(0.1, 0.1, 0.1, 0.05))
    if toggles.get("train_to_tensor", True):
        train_ops.append(transforms.ToTensor())
    if toggles.get("train_normalize", True):
        train_ops.append(normalize)
    if toggles.get("train_random_erasing", False):
        train_ops.append(
            transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0)
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


def evaluate(model: nn.Module, dl: DataLoader, device: str) -> float:
    """计算 top-1 准确率。"""
    model.eval()
    correct = 0
    total = 0
    with torch.inference_mode():
        for batch_x, batch_y in dl:
            inputs = batch_x.to(device, non_blocking=True)
            targets = batch_y.to(device, non_blocking=True)
            logits = model(inputs)
            pred = logits.argmax(1)
            correct += (pred == targets).sum().item()
            total += targets.numel()
    return correct / max(1, total)


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
) -> None:
    """单轮训练循环，包含 AMP 和实时吞吐量报告。"""
    model.train()
    start = perf_counter()
    for i, (batch_x, batch_y) in enumerate(dl, 1):
        inputs = batch_x.to(device, non_blocking=True).to(
            memory_format=torch.channels_last,
        )
        targets = batch_y.to(device, non_blocking=True)
        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type="cuda", enabled=use_cuda_amp):
            loss = criterion(model(inputs), targets)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()

        elapsed = perf_counter() - start
        seen = min(i * dl.batch_size, len(dl.dataset))
        ips = seen / max(1e-6, elapsed)
        progress.update(
            task,
            advance=1,
            description=f"train | loss={loss.item():.4f} | {ips:.0f} img/s",
        )


def main() -> None:  # noqa: PLR0915
    """入口点：设备设置、数据、预热、微调、保存权重。"""
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
    lr = env_float("LR", DEFAULT_LR)
    weight_decay = env_float("WEIGHT_DECAY", DEFAULT_WEIGHT_DECAY)

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

    model = EfficientFormerV2(num_classes=num_classes)
    model = load_efficientformer_v2_pretrained(model, num_classes)
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
    warmup_done = env.resume_checkpoint is not None

    with progress:
        if not warmup_done:
            for name, param in model.named_parameters():
                param.requires_grad = ("head" in name) or ("classifier" in name)

            head_params = [p for p in model.parameters() if p.requires_grad]
            warm_opt = optim.AdamW(head_params, lr=3e-4, weight_decay=5e-2)

            warmup_task = progress.add_task("warmup", total=len(train_dl), extra="")
            console.print("[bold]Warmup (head only)[/]")
            start = perf_counter()

            model.train()
            for i, (batch_x, batch_y) in enumerate(train_dl, 1):
                inputs = batch_x.to(device, non_blocking=True).to(
                    memory_format=torch.channels_last,
                )
                targets = batch_y.to(device, non_blocking=True)
                warm_opt.zero_grad(set_to_none=True)
                with torch.amp.autocast(device_type="cuda", enabled=use_cuda):
                    loss = criterion(model(inputs), targets)
                scaler.scale(loss).backward()
                scaler.step(warm_opt)
                scaler.update()

                elapsed = perf_counter() - start
                seen = min(i * train_dl.batch_size, len(train_dl.dataset))
                ips = seen / max(1e-6, elapsed)
                progress.update(
                    warmup_task,
                    advance=1,
                    description=f"warmup | loss={loss.item():.4f}",
                    extra=f"{ips:.0f} img/s",
                )

            best_val_acc = evaluate(model, val_dl, device)
            console.print(f"[bold cyan]warmup[/] | val_acc={best_val_acc:.4f}")
            best_epoch = 0
            warmup_done = True

        for param in model.parameters():
            param.requires_grad = False
        for name, param in model.named_parameters():
            if any(key in name for key in UNFREEZE_KEYS):
                param.requires_grad = True

        opt = optim.AdamW(
            (p for p in model.parameters() if p.requires_grad),
            lr=lr,
            weight_decay=weight_decay,
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
            console.print(
                f"[bold green]Resumed[/] from epoch {start_epoch} using {env.resume_checkpoint}",
            )

        for epoch in range(start_epoch + 1, epochs + 1):
            task = progress.add_task(f"epoch {epoch}", total=len(train_dl), extra="")
            train_one_epoch(
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
            val_acc = evaluate(model, val_dl, device)
            console.print(f"[bold cyan]epoch {epoch}[/] | val_acc={val_acc:.4f}")

            improved = val_acc > best_val_acc + 1e-4
            if improved:
                best_val_acc = val_acc
                best_epoch = epoch

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

    console.print(f"[bold green]Best weights saved →[/] {env.best_weights_path.resolve()}")
    console.print(
        f"[bold green]Best checkpoint saved →[/] {env.best_checkpoint_path.resolve()}",
    )


if __name__ == "__main__":
    main()

