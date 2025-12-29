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


class PatchEmbed(nn.Module):
    """将图像分割成补丁并嵌入。
    
    使用卷积层实现，比标准的线性投影更高效。
    """

    def __init__(self, img_size: int = 224, patch_size: int = 4, in_channels: int = 3, embed_dim: int = 96) -> None:
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2

        # 使用卷积实现补丁嵌入（更高效）
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播。
        
        参数:
            x: 输入张量，形状为 (B, 3, H, W)
            
        返回:
            嵌入张量，形状为 (B, num_patches, embed_dim)
        """
        x = self.proj(x)  # (B, embed_dim, H', W')
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # (B, H'*W', embed_dim)
        return x


class WindowAttention(nn.Module):
    """窗口注意力机制（FasterViT 的核心）。
    
    将特征图分成窗口，在每个窗口内计算自注意力，减少计算复杂度。
    """

    def __init__(self, dim: int, window_size: int = 7, num_heads: int = 3, qkv_bias: bool = True) -> None:
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        """前向传播。
        
        参数:
            x: 输入张量，形状为 (B, H*W, dim)
            H, W: 特征图的高度和宽度
            
        返回:
            输出张量，形状为 (B, H*W, dim)
        """
        B, N, C = x.shape
        # 重新整形为 (B, H, W, C)
        x = x.reshape(B, H, W, C)

        # 计算窗口数量
        num_windows_h = H // self.window_size
        num_windows_w = W // self.window_size

        # 将特征图分割成窗口
        # (B, H, W, C) -> (B * num_windows, window_size, window_size, C)
        x_windows = x.reshape(
            B,
            num_windows_h,
            self.window_size,
            num_windows_w,
            self.window_size,
            C,
        ).permute(0, 1, 3, 2, 4, 5).reshape(-1, self.window_size * self.window_size, C)

        # 计算注意力
        B_w = x_windows.shape[0]
        qkv = self.qkv(x_windows).reshape(B_w, -1, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        x_windows = (attn @ v).transpose(1, 2).reshape(B_w, -1, C)
        x_windows = self.proj(x_windows)

        # 重新组合窗口
        x = x_windows.reshape(
            B,
            num_windows_h,
            num_windows_w,
            self.window_size,
            self.window_size,
            C,
        ).permute(0, 1, 3, 2, 4, 5).reshape(B, H, W, C)

        # 展平回序列
        x = x.reshape(B, N, C)
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


class FasterViTBlock(nn.Module):
    """FasterViT 块。
    
    包含窗口注意力和 MLP，使用残差连接。
    """

    def __init__(
        self,
        dim: int,
        window_size: int = 7,
        num_heads: int = 3,
        mlp_ratio: float = 4.0,
        drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, window_size, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(dim, mlp_hidden_dim, drop=drop)

    def forward(self, x: torch.Tensor, H: int, W: int) -> torch.Tensor:
        """前向传播。"""
        x = x + self.attn(self.norm1(x), H, W)
        x = x + self.mlp(self.norm2(x))
        return x


class FasterViT(nn.Module):
    """FasterViT 从零实现。
    
    FasterViT 是一个高效的 Vision Transformer，使用窗口注意力机制
    来减少计算复杂度，同时保持全局感受野。
    """

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 4,
        in_channels: int = 3,
        embed_dims: list[int] | None = None,
        num_heads: list[int] | None = None,
        window_sizes: list[int] | None = None,
        depths: list[int] | None = None,
        mlp_ratio: float = 4.0,
        num_classes: int = 2,
        drop_rate: float = 0.0,
    ) -> None:
        """初始化 FasterViT。
        
        参数:
            img_size: 输入图像大小
            patch_size: 补丁大小
            in_channels: 输入通道数
            embed_dims: 各阶段的嵌入维度列表
            num_heads: 各阶段的注意力头数列表
            window_sizes: 各阶段的窗口大小列表
            depths: 各阶段的块数量列表
            mlp_ratio: MLP 的扩展比
            num_classes: 输出类别数
            drop_rate: Dropout 率
        """
        super().__init__()

        # FasterViT-2 的默认配置
        if embed_dims is None:
            embed_dims = [96, 192, 384, 768]
        if num_heads is None:
            num_heads = [3, 6, 12, 24]
        if window_sizes is None:
            window_sizes = [7, 7, 7, 7]
        if depths is None:
            depths = [2, 2, 6, 2]

        self.num_stages = len(embed_dims)
        self.embed_dims = embed_dims

        # 补丁嵌入
        self.patch_embed = PatchEmbed(img_size, patch_size, in_channels, embed_dims[0])

        # 构建各阶段
        self.stages = nn.ModuleList()
        for i in range(self.num_stages):
            stage = nn.ModuleList()
            for _ in range(depths[i]):
                stage.append(
                    FasterViTBlock(
                        embed_dims[i],
                        window_sizes[i],
                        num_heads[i],
                        mlp_ratio,
                        drop_rate,
                    )
                )
            self.stages.append(stage)

        # 阶段间的下采样（如果需要）
        self.downsample_layers = nn.ModuleList()
        for i in range(self.num_stages - 1):
            self.downsample_layers.append(
                nn.Sequential(
                    nn.LayerNorm(embed_dims[i]),
                    nn.Linear(embed_dims[i], embed_dims[i + 1]),
                )
            )

        # 分类头
        self.norm = nn.LayerNorm(embed_dims[-1])
        self.head = nn.Linear(embed_dims[-1], num_classes)

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """初始化模型权重。"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播。
        
        参数:
            x: 输入张量，形状为 (B, 3, H, W)
            
        返回:
            输出 logits，形状为 (B, num_classes)
        """
        B = x.shape[0]
        H = W = self.patch_embed.img_size // self.patch_embed.patch_size

        # 补丁嵌入
        x = self.patch_embed(x)  # (B, H*W, embed_dim[0])

        # 通过各阶段
        for i, stage in enumerate(self.stages):
            # 计算当前阶段的特征图大小
            if i > 0:
                H = W = H // 2  # 下采样

            # 通过该阶段的所有块
            for block in stage:
                x = block(x, H, W)

            # 下采样到下一阶段（除了最后一个阶段）
            if i < self.num_stages - 1:
                # 重新整形为特征图
                x = x.reshape(B, H, W, -1)
                # 下采样（简化：使用平均池化）
                x = x.permute(0, 3, 1, 2)  # (B, C, H, W)
                x = nn.functional.avg_pool2d(x, 2, 2)  # (B, C, H/2, W/2)
                x = x.permute(0, 2, 3, 1)  # (B, H/2, W/2, C)
                x = x.reshape(B, -1, x.shape[-1])  # (B, H/2*W/2, C)
                # 投影到下一阶段的维度
                x = self.downsample_layers[i](x)

        # 全局平均池化
        x = self.norm(x)  # (B, N, C)
        x = x.mean(dim=1)  # (B, C)

        # 分类
        x = self.head(x)  # (B, num_classes)

        return x


def load_fastervit_pretrained(model: nn.Module, num_classes: int) -> nn.Module:
    """从 timm 或 fastervit 库加载 ImageNet 预训练权重到自定义 FasterViT。
    
    此函数尝试从 timm 或 fastervit 库加载 FasterViT-2 的预训练权重。
    由于架构可能不完全匹配，会尝试加载兼容的权重。
    
    参数:
        model: 自定义 FasterViT 模型实例
        num_classes: 类别数（分类器将被替换，不加载）
    
    返回:
        加载了预训练权重的模型（如果可用）
    """
    try:
        # 首先尝试从 timm 加载
        try:
            import timm

            pretrained = timm.create_model("faster_vit_2_224", pretrained=True, num_classes=1000)
            pretrained_dict = pretrained.state_dict()
            source = "timm"
        except Exception:
            # 如果 timm 不可用，尝试从 fastervit 库加载
            try:
                from fastervit import create_model

                pretrained = create_model("faster_vit_2_224", pretrained=True)
                pretrained_dict = pretrained.state_dict()
                source = "fastervit"
            except Exception:
                raise ImportError("无法从 timm 或 fastervit 加载预训练模型")

        model_dict = model.state_dict()

        # 移除分类器权重
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if "head" not in k and "classifier" not in k}

        # 创建权重映射字典
        mapped_dict = {}

        # 尝试映射权重
        for pretrained_key, pretrained_weight in pretrained_dict.items():
            # 尝试直接匹配
            if pretrained_key in model_dict and pretrained_weight.shape == model_dict[pretrained_key].shape:
                mapped_dict[pretrained_key] = pretrained_weight
                continue

            # 尝试部分匹配（处理命名差异）
            # FasterViT 的命名可能有所不同
            for model_key in model_dict.keys():
                # 检查是否是同一类型的层
                pretrained_parts = pretrained_key.split(".")
                model_parts = model_key.split(".")

                # 如果最后几层名称匹配，且形状相同
                if len(pretrained_parts) >= 2 and len(model_parts) >= 2:
                    if (
                        pretrained_parts[-2:] == model_parts[-2:]
                        and pretrained_weight.shape == model_dict[model_key].shape
                    ):
                        mapped_dict[model_key] = pretrained_weight
                        break

        # 更新模型权重
        model_dict.update(mapped_dict)
        model.load_state_dict(model_dict, strict=False)

        print(f"[FasterViT] 从 {source} 成功加载 {len(mapped_dict)}/{len(pretrained_dict)} 个预训练权重层")

    except Exception as e:
        print(f"[FasterViT] 预训练权重加载失败: {e}，使用随机初始化")

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

BEST_WEIGHTS_NAME: str = "FasterVitFromScratch.pth"
BEST_CKPT_NAME: str = "best.ckpt"
LATEST_CKPT_NAME: str = "latest.ckpt"

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


def evaluate(model: nn.Module, dl: DataLoader, device: str) -> EvalResult:
    """计算 top-1 准确率。"""
    model.eval()
    correct = 0
    total = 0
    with torch.inference_mode():
        for batch_x, batch_y in dl:
            inputs = batch_x.to(device, non_blocking=True).to(
                memory_format=torch.channels_last,
            )
            targets = batch_y.to(device, non_blocking=True)
            logits = model(inputs)
            pred = logits.argmax(1)
            correct += (pred == targets).sum().item()
            total += targets.numel()
    acc = correct / max(1, total)
    return EvalResult(acc=acc, total=total, correct=correct)


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
) -> None:
    """单轮训练循环，包含 AMP 和实时吞吐量报告。"""
    model.train()
    start = perf_counter()
    opt.zero_grad(set_to_none=True)

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

    model = FasterViT(img_size=img_size, num_classes=num_classes)
    model = load_fastervit_pretrained(model, num_classes)
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
            for param in model.parameters():
                param.requires_grad = False
            for name, param in model.named_parameters():
                if "head" in name:
                    param.requires_grad = True

            head_params = [p for p in model.parameters() if p.requires_grad]
            warm_opt = optim.AdamW(head_params, lr=HEAD_LR, weight_decay=HEAD_WD)

            warm_task = progress.add_task(
                "warmup (head only)",
                total=len(train_dl),
                extra="",
            )
            console.print("[bold]Warmup (head only)[/]")
            train_one_epoch(
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

            res = evaluate(model, val_dl, device)
            console.print(
                f"[bold cyan]warmup[/] | val_acc={res.acc:.4f} ({res.correct}/{res.total})",
            )
            best_val_acc = res.acc
            best_epoch = 0
            warmup_done = True

        for param in model.parameters():
            param.requires_grad = True

        ft_batch_size = 32
        effective_batch = 128
        accum_steps_ft = max(1, effective_batch // ft_batch_size)
        console.print(
            f"[bold]Fine-tune[/]: bs={ft_batch_size}, accum_steps={accum_steps_ft} "
            f"(effective ≈ {ft_batch_size * accum_steps_ft})",
        )

        train_dl_ft = DataLoader(
            train_dl.dataset,
            batch_size=ft_batch_size,
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
            train_one_epoch(
                model=model,
                dl=train_dl_ft,
                opt=opt,
                scaler=scaler,
                criterion=criterion,
                device=device,
                use_cuda_amp=use_cuda,
                progress=progress,
                task=task,
                accum_steps=accum_steps_ft,
            )
            scheduler.step()

            res = evaluate(model, val_dl, device)
            console.print(
                f"[bold cyan]epoch {epoch}[/] | val_acc={res.acc:.4f} "
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
                    f"[bold green]↑ new best[/] val_acc={best_val_acc:.4f} "
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

