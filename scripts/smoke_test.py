"""
Smoke test: verify the DCAE model forward pass works on dummy input.

Run from the DCAE repo root (after forking + cloning):
    python scripts/smoke_test.py

Pass criteria printed at the end.
"""
import sys

print("Checking imports...", flush=True)
try:
    import torch
    print(f"  torch {torch.__version__}")
except ImportError:
    sys.exit("torch not installed. Run: pip install torch torchvision")

try:
    import compressai
    print(f"  compressai {compressai.__version__}")
except ImportError:
    sys.exit("compressai not installed. Run: pip install compressai==1.2.6")

try:
    import timm
    print(f"  timm {timm.__version__}")
except ImportError:
    sys.exit("timm not installed. Run: pip install timm")

try:
    import einops
    print(f"  einops {einops.__version__}")
except ImportError:
    sys.exit("einops not installed. Run: pip install einops")

print("\nLoading DCAE model...", flush=True)
sys.path.insert(0, ".")
try:
    from models.dcae import DCAE
except Exception as e:
    sys.exit(f"Failed to import DCAE: {e}")

print("Building DCAE(N=192, M=320)...", flush=True)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"  device: {device}")
net = DCAE(N=192, M=320).to(device).eval()

n_params = sum(p.numel() for p in net.parameters()) / 1e6
print(f"  parameters: {n_params:.1f} M")

print("\nRunning forward pass on 1×3×256×256 dummy input...", flush=True)
x = torch.rand(1, 3, 256, 256, device=device)
with torch.no_grad():
    out = net(x)

x_hat = out["x_hat"]
y_like = out["likelihoods"]["y"]
z_like = out["likelihoods"]["z"]

print(f"  x_hat shape:        {list(x_hat.shape)}")
print(f"  y_likelihoods shape:{list(y_like.shape)}")
print(f"  z_likelihoods shape:{list(z_like.shape)}")

checks = [
    ("x_hat is [1,3,256,256]", list(x_hat.shape) == [1, 3, 256, 256]),
    ("x_hat values finite (clamped in eval only)", not torch.isinf(x_hat).any().item()),
    ("y_likelihoods 4-dim",    len(y_like.shape) == 4),
    ("z_likelihoods 4-dim",    len(z_like.shape) == 4),
    ("no NaNs in x_hat",       not torch.isnan(x_hat).any().item()),
]

print("\n--- Results ---")
all_pass = True
for name, ok in checks:
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] {name}")
    if not ok:
        all_pass = False

if all_pass:
    print("\nAll checks passed. Model is working correctly.")
else:
    print("\nSome checks FAILED. Check your installation.")
    sys.exit(1)
