"""
Download all 24 Kodak lossless true-colour images to a local folder.

Usage:
    python scripts/download_kodak.py --out_dir datasets/kodak
"""
import argparse
import os
import urllib.request
from pathlib import Path


KODAK_BASE = "https://r0k.us/graphics/kodak/kodak/kodim{:02d}.png"
NUM_IMAGES = 24


def download(out_dir: Path, verbose: bool = True):
    out_dir.mkdir(parents=True, exist_ok=True)
    for i in range(1, NUM_IMAGES + 1):
        url = KODAK_BASE.format(i)
        dst = out_dir / f"kodim{i:02d}.png"
        if dst.exists():
            if verbose:
                print(f"  already exists: {dst.name}")
            continue
        if verbose:
            print(f"  downloading {url} ...", end=" ", flush=True)
        try:
            urllib.request.urlretrieve(url, dst)
            if verbose:
                size_kb = os.path.getsize(dst) // 1024
                print(f"done ({size_kb} KB)")
        except Exception as e:
            print(f"FAILED: {e}")
    print(f"\nKodak images saved to: {out_dir}  ({len(list(out_dir.glob('*.png')))} files)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", default="datasets/kodak",
                        help="Destination folder (default: datasets/kodak)")
    args = parser.parse_args()
    download(Path(args.out_dir))


if __name__ == "__main__":
    main()
