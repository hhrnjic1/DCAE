"""
Build a class-balanced ImageNet subset for DCAE fine-tuning.

Produces the flat folder layout that compressai.datasets.ImageFolder expects:
    <out_dir>/train/  – flat .jpg files, up to --per_class per ImageNet class
    <out_dir>/test/   – flat .jpg files, 1 000 reserved for the in-loop test split

Usage (HuggingFace streaming, recommended):
    python scripts/build_imagenet_subset.py \
        --source hf \
        --out_dir /content/imagenet_dcae \
        --per_class 30 \
        --min_side 256 \
        --max_side 512

Usage (local unpacked tarball):
    python scripts/build_imagenet_subset.py \
        --source local \
        --root /path/to/imagenet/train \
        --out_dir /content/imagenet_dcae \
        --per_class 30
"""
import argparse
import os
import random
from pathlib import Path

from PIL import Image


# ──────────────────────────────────────────────────────────────
# helpers
# ──────────────────────────────────────────────────────────────

def passes_size(img: Image.Image, min_side: int) -> bool:
    return min(img.width, img.height) >= min_side


def resize_long_edge(img: Image.Image, max_side: int) -> Image.Image:
    if max(img.width, img.height) <= max_side:
        return img
    if img.width >= img.height:
        new_w = max_side
        new_h = round(img.height * max_side / img.width)
    else:
        new_h = max_side
        new_w = round(img.width * max_side / img.height)
    return img.resize((new_w, new_h), Image.LANCZOS)


def save_flat(img: Image.Image, out_dir: Path, counter: int) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    dst = out_dir / f"{counter:07d}.jpg"
    img.convert("RGB").save(dst, "JPEG", quality=95)
    return dst


# ──────────────────────────────────────────────────────────────
# sources
# ──────────────────────────────────────────────────────────────

def iter_hf(split: str = "train"):
    """Stream images from HuggingFace imagenet-1k (requires HF login + license accept)."""
    try:
        from datasets import load_dataset
    except ImportError:
        raise SystemExit("Install with: pip install datasets")
    ds = load_dataset("imagenet-1k", split=split, streaming=True, trust_remote_code=True)
    for sample in ds:
        pil = sample["image"]
        label = sample["label"]
        if not isinstance(pil, Image.Image):
            continue
        yield pil, label


def iter_local(root: str):
    """Walk a local tarball-unpacked ImageNet train tree (class folders)."""
    root_path = Path(root)
    classes = sorted(p for p in root_path.iterdir() if p.is_dir())
    for cls_dir in classes:
        cls_id = cls_dir.name
        files = list(cls_dir.glob("*.JPEG")) + list(cls_dir.glob("*.jpg")) + \
                list(cls_dir.glob("*.png"))
        random.shuffle(files)
        for f in files:
            try:
                yield Image.open(f), cls_id
            except Exception:
                continue


# ──────────────────────────────────────────────────────────────
# main
# ──────────────────────────────────────────────────────────────

def build(args):
    out_train = Path(args.out_dir) / "train"
    out_test = Path(args.out_dir) / "test"
    out_train.mkdir(parents=True, exist_ok=True)
    out_test.mkdir(parents=True, exist_ok=True)

    per_class: dict[object, int] = {}
    train_count = 0
    test_count = 0
    target_train = args.per_class * 1000
    target_test = args.test_size

    print(f"Target: {target_train} train + {target_test} test images, "
          f"min_side={args.min_side}, max_side={args.max_side}")

    if args.source == "hf":
        stream = iter_hf("train")
    else:
        if not args.root:
            raise SystemExit("--root is required for --source local")
        stream = iter_local(args.root)

    for img, label in stream:
        done_train = train_count >= target_train
        done_test = test_count >= target_test
        if done_train and done_test:
            break

        if not passes_size(img, args.min_side):
            continue

        img = resize_long_edge(img, args.max_side)

        # reserve early samples for test (one per 30 train)
        use_for_test = (not done_test) and (test_count < target_test) and \
                       (random.random() < target_test / max(target_train, 1))

        if use_for_test:
            save_flat(img, out_test, test_count)
            test_count += 1
        elif not done_train:
            count_for_class = per_class.get(label, 0)
            if count_for_class >= args.per_class:
                continue
            save_flat(img, out_train, train_count)
            per_class[label] = count_for_class + 1
            train_count += 1

        if (train_count + test_count) % 1000 == 0:
            print(f"  saved {train_count} train + {test_count} test ...", flush=True)

    print(f"\nDone. train={train_count}, test={test_count}")
    print(f"  → {out_train}")
    print(f"  → {out_test}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", choices=["hf", "local"], default="hf",
                        help="hf = HuggingFace streaming; local = unpacked tarball")
    parser.add_argument("--root", default=None,
                        help="Path to ImageNet train/ tree (for --source local)")
    parser.add_argument("--out_dir", required=True,
                        help="Output root (will contain train/ and test/)")
    parser.add_argument("--per_class", type=int, default=30,
                        help="Max images per class in train split (default 30)")
    parser.add_argument("--test_size", type=int, default=1000,
                        help="Number of images to reserve for test split (default 1000)")
    parser.add_argument("--min_side", type=int, default=256,
                        help="Skip images with min(W,H) < this (default 256)")
    parser.add_argument("--max_side", type=int, default=512,
                        help="Resize long edge to this (default 512)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    random.seed(args.seed)
    build(args)


if __name__ == "__main__":
    main()
