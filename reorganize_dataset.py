#!/usr/bin/env python3
"""Reorganize dataset into ImageFolder format with train/val splits.

This script:
1. Reads images from train/ and test/ directories
2. Extracts class labels from filenames (format: {class}_big (...).png)
3. Creates class-based directory structure
4. Splits train into train/val (default 80/20)
5. Organizes test set by class
"""

from __future__ import annotations

import argparse
import re
import shutil
from collections import defaultdict
from pathlib import Path
from random import Random

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, MofNCompleteColumn

console = Console()

# Supported image extensions
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG"}


def extract_class_from_filename(filename: str) -> str | None:
    """Extract class label from filename.
    
    Expected format: {class}_big (...).png
    Returns class label as string, or None if pattern doesn't match.
    """
    match = re.match(r"^(\d+)_", filename)
    if match:
        return match.group(1)
    return None


def collect_images_by_class(source_dir: Path) -> dict[str, list[Path]]:
    """Collect all images from source directory, grouped by class."""
    images_by_class: dict[str, list[Path]] = defaultdict(list)
    
    if not source_dir.exists():
        console.print(f"[yellow]Directory not found:[/] {source_dir}")
        return images_by_class
    
    image_files = [
        f for f in source_dir.iterdir()
        if f.is_file() and f.suffix in IMAGE_EXTENSIONS
    ]
    
    for img_path in image_files:
        class_label = extract_class_from_filename(img_path.name)
        if class_label is None:
            console.print(f"[yellow]Skipping file with unknown class pattern:[/] {img_path.name}")
            continue
        images_by_class[class_label].append(img_path)
    
    return images_by_class


def split_train_val(
    images_by_class: dict[str, list[Path]],
    val_ratio: float = 0.2,
    seed: int = 42,
) -> tuple[dict[str, list[Path]], dict[str, list[Path]]]:
    """Split images into train and validation sets, maintaining class balance."""
    train_images: dict[str, list[Path]] = defaultdict(list)
    val_images: dict[str, list[Path]] = defaultdict(list)
    
    rng = Random(seed)
    
    for class_label, images in images_by_class.items():
        # Shuffle to ensure random split
        shuffled = images.copy()
        rng.shuffle(shuffled)
        
        # Calculate split point
        val_count = max(1, int(len(shuffled) * val_ratio))
        
        # Split
        val_images[class_label] = shuffled[:val_count]
        train_images[class_label] = shuffled[val_count:]
    
    return train_images, val_images


def create_directory_structure(base_dir: Path, splits: list[str], classes: list[str]) -> None:
    """Create directory structure for ImageFolder format."""
    for split in splits:
        for class_label in classes:
            dir_path = base_dir / split / f"class{class_label}"
            dir_path.mkdir(parents=True, exist_ok=True)


def copy_images(
    images_by_class: dict[str, list[Path]],
    target_dir: Path,
    split_name: str,
    progress: Progress | None = None,
    move: bool = False,
) -> int:
    """Copy or move images to target directory structure."""
    total_processed = 0
    
    for class_label, images in images_by_class.items():
        class_dir = target_dir / split_name / f"class{class_label}"
        class_dir.mkdir(parents=True, exist_ok=True)
        
        for img_path in images:
            target_path = class_dir / img_path.name
            if target_path.exists():
                console.print(f"[yellow]File already exists, skipping:[/] {target_path}")
                continue
            
            if move:
                shutil.move(str(img_path), str(target_path))
            else:
                shutil.copy2(img_path, target_path)
            total_processed += 1
            
            if progress:
                progress.advance(progress.tasks[0].id)
    
    return total_processed


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reorganize dataset into ImageFolder format with train/val splits"
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("/home/tjj/nas/dataset/biggan/data_big"),
        help="Source dataset directory (default: /home/tjj/nas/dataset/biggan/data_big)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output directory (default: same as source, creates backup)",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.2,
        help="Ratio of training data to use for validation (default: 0.2)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for train/val split (default: 42)",
    )
    parser.add_argument(
        "--move",
        action="store_true",
        help="Move files instead of copying (default: copy)",
    )
    args = parser.parse_args()
    
    source_dir = args.source.resolve()
    if args.output:
        output_dir = args.output.resolve()
    else:
        # Use same directory, but create organized structure
        output_dir = source_dir
    
    console.print(f"[bold]Source directory:[/] {source_dir}")
    console.print(f"[bold]Output directory:[/] {output_dir}")
    console.print(f"[bold]Validation ratio:[/] {args.val_ratio}")
    
    # Check source directory
    train_source = source_dir / "train"
    test_source = source_dir / "test"
    
    if not train_source.exists():
        console.print(f"[bold red]Error:[/] Train directory not found: {train_source}")
        raise SystemExit(1)
    
    # Collect images from train set
    console.print("\n[bold]Collecting training images...[/]")
    train_images_by_class = collect_images_by_class(train_source)
    
    if not train_images_by_class:
        console.print("[bold red]Error:[/] No images found in train directory")
        raise SystemExit(1)
    
    # Show class distribution
    console.print("\n[bold]Training set class distribution:[/]")
    total_train = 0
    for class_label in sorted(train_images_by_class.keys()):
        count = len(train_images_by_class[class_label])
        total_train += count
        console.print(f"  Class {class_label}: {count} images")
    console.print(f"  [bold]Total:[/] {total_train} images")
    
    # Split train into train/val
    console.print(f"\n[bold]Splitting train set (val_ratio={args.val_ratio})...[/]")
    train_split, val_split = split_train_val(train_images_by_class, args.val_ratio, args.seed)
    
    console.print("\n[bold]Split results:[/]")
    for class_label in sorted(train_split.keys()):
        train_count = len(train_split[class_label])
        val_count = len(val_split[class_label])
        console.print(f"  Class {class_label}: train={train_count}, val={val_count}")
    
    # Collect test images
    test_images_by_class = {}
    if test_source.exists():
        console.print("\n[bold]Collecting test images...[/]")
        test_images_by_class = collect_images_by_class(test_source)
        if test_images_by_class:
            console.print("\n[bold]Test set class distribution:[/]")
            total_test = 0
            for class_label in sorted(test_images_by_class.keys()):
                count = len(test_images_by_class[class_label])
                total_test += count
                console.print(f"  Class {class_label}: {count} images")
            console.print(f"  [bold]Total:[/] {total_test} images")
    
    # Get all classes
    all_classes = sorted(set(train_images_by_class.keys()) | set(test_images_by_class.keys()))
    
    # Create directory structure
    console.print("\n[bold]Creating directory structure...[/]")
    splits = ["train", "val"]
    if test_images_by_class:
        splits.append("test")
    create_directory_structure(output_dir, splits, all_classes)
    
    # Copy/move images
    operation = "Moving" if args.move else "Copying"
    console.print(f"\n[bold]{operation} images...[/]")
    
    total_files = (
        sum(len(imgs) for imgs in train_split.values())
        + sum(len(imgs) for imgs in val_split.values())
        + sum(len(imgs) for imgs in test_images_by_class.values())
    )
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        console=console,
    ) as progress:
        task = progress.add_task(f"{operation.lower()} images", total=total_files)
        
        # Copy/move train set
        train_count = copy_images(train_split, output_dir, "train", progress, move=args.move)
        action = "Moved" if args.move else "Copied"
        console.print(f"[green]✓[/] {action} {train_count} training images")
        
        # Copy/move val set
        val_count = copy_images(val_split, output_dir, "val", progress, move=args.move)
        console.print(f"[green]✓[/] {action} {val_count} validation images")
        
        # Copy/move test set
        if test_images_by_class:
            test_count = copy_images(test_images_by_class, output_dir, "test", progress, move=args.move)
            console.print(f"[green]✓[/] {action} {test_count} test images")
    
    console.print("\n[bold green]✓ Dataset reorganization complete![/]")
    console.print(f"\n[bold]New structure:[/]")
    console.print(f"  {output_dir}/train/class0/")
    console.print(f"  {output_dir}/train/class1/")
    console.print(f"  {output_dir}/val/class0/")
    console.print(f"  {output_dir}/val/class1/")
    if test_images_by_class:
        console.print(f"  {output_dir}/test/class0/")
        console.print(f"  {output_dir}/test/class1/")


if __name__ == "__main__":
    main()

