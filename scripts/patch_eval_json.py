"""
Add --results_json flag to eval.py so BD-Rate can be computed offline.

Run ONCE from inside your cloned DCAE repo root:
    python scripts/patch_eval_json.py

After patching, eval.py accepts:
    python eval.py --checkpoint ckpt.pth --data datasets/kodak --cuda --real \
                   --results_json results/kodak_mse_0.0067.json

The JSON written has keys:
    {"lambda": float|null, "bpp": float, "psnr": float, "msssim": float,
     "enc_ms": float, "dec_ms": float}
"""
import re
import sys
from pathlib import Path

EVAL_PY = Path("eval.py")

if not EVAL_PY.exists():
    sys.exit("Run this script from the DCAE repo root (eval.py not found here).")

src = EVAL_PY.read_text()

if "results_json" in src:
    print("eval.py already contains results_json — skipping (already patched).")
    sys.exit(0)

# ── 1. add json import ───────────────────────────────────────
if "import json" not in src:
    src = "import json\n" + src

# ── 2. add --results_json argparse flag ──────────────────────
json_flag = '''
    parser.add_argument(
        "--results_json", default=None,
        help="If set, write aggregated metrics to this JSON file (for BD-Rate computation)"
    )
    parser.add_argument(
        "--lmbda", type=float, default=None,
        help="Lambda value to record in results_json (optional, for bookkeeping)"
    )'''

# insert after --real flag (or --save_path)
for anchor_str in ['parser.add_argument(\n        "--real"', '"--real"', '"--save_path"']:
    idx = src.find(anchor_str)
    if idx != -1:
        close_idx = src.find("\n    )", idx)
        if close_idx != -1:
            insert_at = close_idx + len("\n    )")
            src = src[:insert_at] + json_flag + src[insert_at:]
            break
else:
    # fallback
    src = src.replace("args = parser.parse_args()", json_flag + "\n    args = parser.parse_args()", 1)

# ── 3. write JSON at the end of main() ──────────────────────
json_write_block = '''
    if args.results_json:
        out = {
            "lambda": args.lmbda,
            "bpp": float(bpp_sum / num_images),
            "psnr": float(psnr_sum / num_images),
            "msssim": float(ms_ssim_sum / num_images),
            "enc_ms": float(enc_time_sum / num_images * 1000),
            "dec_ms": float(dec_time_sum / num_images * 1000),
        }
        Path(args.results_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.results_json, "w") as _f:
            json.dump(out, _f, indent=2)
        print(f"Results saved to {args.results_json}")
'''

# Anchor: the final print statement that shows averages
# Look for something like "print(" near end of file
final_print_patterns = [
    r'(\s+print\(f?".*(?:avg|average|PSNR|bpp).*"\))',
    r'(\s+print\(".*(?:avg|average|PSNR|bpp).*"\))',
]
inserted = False
for pat in final_print_patterns:
    matches = list(re.finditer(pat, src, re.IGNORECASE))
    if matches:
        last_match = matches[-1]
        insert_at = last_match.end()
        src = src[:insert_at] + "\n" + json_write_block + src[insert_at:]
        inserted = True
        break

if not inserted:
    # Append before if __name__ == "__main__"
    anchor = 'if __name__ == "__main__"'
    idx = src.find(anchor)
    if idx != -1:
        src = src[:idx] + json_write_block + "\n\n" + src[idx:]
    else:
        src += "\n" + json_write_block

# ── 4. also ensure Path is imported ─────────────────────────
if "from pathlib import Path" not in src and "import pathlib" not in src:
    src = "from pathlib import Path\n" + src

EVAL_PY.write_text(src)
print("eval.py patched successfully.")
print("  + --results_json flag")
print("  + --lmbda flag (for bookkeeping in the JSON)")
print("  + writes JSON with bpp, psnr, msssim, enc_ms, dec_ms at end of eval run")
