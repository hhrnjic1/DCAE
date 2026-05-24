"""
Apply AMP (mixed-precision) and checkpointing-frequency patches to train.py.

Run ONCE from inside your cloned DCAE repo root:
    python scripts/patch_train_amp.py

What it adds to train.py:
  1. --amp flag (torch.cuda.amp.GradScaler)
  2. --save_interval flag (checkpoint every N iterations, default 500)

These changes are backward-compatible: without --amp the behaviour is identical to upstream.
"""
import re
import sys
from pathlib import Path

TRAIN_PY = Path("train.py")

if not TRAIN_PY.exists():
    sys.exit("Run this script from the DCAE repo root (train.py not found here).")

src = TRAIN_PY.read_text()

# ── guard: already patched? ──────────────────────────────────
if "GradScaler" in src:
    print("train.py already contains GradScaler — skipping (already patched).")
    sys.exit(0)

# ── 1. add AMP import after existing torch imports ───────────
amp_import = "from torch.cuda.amp import GradScaler, autocast\n"
# insert right after "import torch" block
src = re.sub(
    r"(import torch\b[^\n]*\n)",
    r"\1" + amp_import,
    src,
    count=1,
)

# ── 2. add --amp and --save_interval argparse flags ──────────
# Find the line that adds --cuda so we can insert right after it
amp_flag_block = '''
    parser.add_argument(
        "--amp", action="store_true", default=False,
        help="Use AMP mixed precision (recommended on T4/Colab)"
    )
    parser.add_argument(
        "--save_interval", type=int, default=500,
        help="Save checkpoint_latest.pth.tar every N iterations (default 500)"
    )'''

# anchor: line that defines --cuda flag
cuda_anchor = 'parser.add_argument(\n        "--cuda"'
if cuda_anchor not in src:
    # fallback: try single-line form
    cuda_anchor = '"--cuda"'
idx = src.find(cuda_anchor)
if idx == -1:
    print("WARNING: could not find --cuda argparse entry; inserting AMP flags at end of parse_args block.")
    src = src.replace("args = parser.parse_args()", amp_flag_block + "\n    args = parser.parse_args()", 1)
else:
    # find the closing ) of the --cuda add_argument call
    close_idx = src.find("\n    )", idx)
    insert_at = close_idx + len("\n    )")
    src = src[:insert_at] + amp_flag_block + src[insert_at:]

# ── 3. create GradScaler when amp flag is set ────────────────
# Insert after "net = net.cuda()" or similar cuda call in main
scaler_init = '''
    scaler = GradScaler(enabled=args.amp)
'''
# anchor: the optimizer zero_grad inside the training loop
zero_grad_anchor = "optimizer.zero_grad()"
if zero_grad_anchor in src:
    # insert scaler creation before the training loop: find "for i, d in" loop
    for_loop_anchor = "for i, d in"
    loop_idx = src.find(for_loop_anchor)
    if loop_idx != -1:
        # insert scaler init just before the loop
        line_start = src.rfind("\n", 0, loop_idx) + 1
        src = src[:line_start] + scaler_init + src[line_start:]

# ── 4. wrap forward pass with autocast ──────────────────────
# Pattern: out_net = net(d) inside training loop
src = re.sub(
    r"(\s+)(out_net = net\(d\))",
    r"\1with autocast(enabled=args.amp):\n\1    \2",
    src,
    count=1,
)

# ── 5. replace loss.backward() with scaler-based backward ───
src = src.replace(
    "loss.backward()",
    "scaler.scale(loss).backward()",
    1,
)
src = re.sub(
    r"(\s+)(optimizer\.step\(\))",
    r"\1scaler.unscale_(optimizer)\n\1torch.nn.utils.clip_grad_norm_("
    r"net.parameters(), args.clip_max_norm)\n\1scaler.step(optimizer)\n\1scaler.update()",
    src,
    count=1,
)
# Remove the original clip_grad_norm_ call if it existed separately
src = re.sub(
    r"\s+torch\.nn\.utils\.clip_grad_norm_\(net\.parameters\(\),\s*args\.clip_max_norm\)\n",
    "\n",
    src,
)

# ── 6. add per-interval checkpointing inside training loop ───
# Insert after the optimizer step block, before aux optimizer block
interval_save = '''
        if args.save and (i + 1) % args.save_interval == 0:
            save_checkpoint(
                {
                    "epoch": epoch,
                    "iter": i,
                    "state_dict": net.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "aux_optimizer": aux_optimizer.state_dict(),
                    "lr_scheduler": lr_scheduler.state_dict(),
                },
                False,
                args.save_path,
            )
'''
# anchor: right before "aux_optimizer.zero_grad()"
aux_zero = "aux_optimizer.zero_grad()"
if aux_zero in src:
    idx = src.find(aux_zero)
    line_start = src.rfind("\n", 0, idx) + 1
    src = src[:line_start] + interval_save + src[line_start:]

TRAIN_PY.write_text(src)
print("train.py patched successfully.")
print("  + --amp flag (AMP mixed precision)")
print("  + --save_interval flag (checkpoint every N iters, default 500)")
print("  + autocast wrapping around forward pass")
print("  + GradScaler-based backward/step")
