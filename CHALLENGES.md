# Engineering Challenges & Resolutions

**Project:** DCAE Image Compression — MMS 2025/26, Topic 3
**Institution:** ETF Sarajevo
**Authors:** Hamza Hrnjić, Safet
**Base repository:** fork of [CVL-UESTC/DCAE](https://github.com/CVL-UESTC/DCAE)

This document records the technical obstacles encountered while porting the original
DCAE reference implementation to a reproducible, single-GPU Google Colab (T4) workflow,
and how each was resolved. It is intended as a companion to the project report.

The original code targeted multi-GPU (DistributedDataParallel) training on local
hardware with a pre-prepared dataset. Adapting it to a free Colab T4 — limited memory,
single GPU, streamed data, and frequent disconnects — surfaced a series of
incompatibilities. They are documented below in the order they were encountered.

---

## Summary

| # | Challenge | Area | Commit |
|---|-----------|------|--------|
| 1 | Kodak benchmark download failed (HTTP redirect) | Data acquisition | `0fe0ba4` |
| 2 | Mixed-precision (AMP) training path was broken | Training | `a396568` |
| 3 | `--results_json` exported incorrect metric values | Evaluation | `787daa1` |
| 4 | Pretrained checkpoints could not be loaded (DDP `module.` prefix) | Checkpoints | `6019f40` |
| 5 | Entropy-model CDF buffers missing on load | Checkpoints | `8de3504` |
| 6 | ImageNet-1k streaming load crashed on current `datasets` | Data acquisition | *(latest)* |
| 7 | Kodak set absent on a clean Colab clone | Reproducibility | *(latest)* |
| 8 | Fine-tune ran zero epochs and saved no checkpoint | Training / Checkpoints | *(latest)* |
| 9 | Stale Colab clone ran outdated `train.py` (#8 fix never reached the runtime) | Reproducibility | *(latest)* |
| 10 | Sub-256 images crashed fine-tune `RandomCrop` mid-epoch | Data acquisition / Training | *(latest)* |

---

## 1. Kodak benchmark download failed silently

**Symptom.** `scripts/download_kodak.py` produced empty or missing image files; the
subsequent Kodak evaluation had no data to read.

**Root cause.** The image source was requested over `http://r0k.us/...`, which responds
with a `301` permanent redirect to `https://`. `urllib.request.urlretrieve` did not
transparently complete the redirect for the file write.

**Resolution.** Pointed `KODAK_BASE` directly at the `https://` endpoint so the 24
benchmark images download in a single request.

```diff
- KODAK_BASE = "http://r0k.us/graphics/kodak/kodak/kodim{:02d}.png"
+ KODAK_BASE = "https://r0k.us/graphics/kodak/kodak/kodim{:02d}.png"
```

---

## 2. Mixed-precision (AMP) training path was broken

**Context.** Half-precision training (`torch.cuda.amp`) is required to fit DCAE within
the T4's 16 GB of memory. The initial patch that introduced the `--amp` flag contained
three defects in `train_one_epoch`.

**Root causes.**
1. The auxiliary (entropy-bottleneck) loss was scaled through the AMP `GradScaler` and
   referenced an undefined `loss` variable (`aux_scaler.scale(loss).backward()`), so the
   quantile-fitting step never ran correctly.
2. Gradient unscaling and clipping were performed in the wrong order relative to
   `scaler.step`/`scaler.update`.
3. `autocast` was imported from the deprecated `torch.cuda.amp` namespace, and the
   interval checkpoint saved the wrong object (`net` instead of the wrapped `model`).

**Resolution.** Rewrote the step so that the **main** loss flows through the GradScaler
under `autocast`, while the **auxiliary** loss is computed and back-propagated in full
precision (fp32) — entropy-bottleneck quantile fitting is numerically sensitive and must
not run in fp16.

```python
with autocast("cuda", enabled=args.amp):
    out_net = model(d)
    out_criterion = criterion(out_net, d)

scaler.scale(out_criterion["loss"]).backward()
if clip_max_norm > 0:
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), clip_max_norm)
scaler.step(optimizer)
scaler.update()

# auxiliary loss kept in fp32
aux_loss = model.module.aux_loss() if torch.cuda.device_count() > 1 else model.aux_loss()
aux_loss.backward()
aux_optimizer.step()
```

The interval checkpoint was also corrected to persist `model.state_dict()`, enabling
disconnect-safe mid-epoch saves to Google Drive.

---

## 3. `--results_json` exported incorrect metric values

**Symptom.** The machine-readable evaluation output (consumed by the BD-Rate stage)
contained metric values that did not match the human-readable console report.

**Root cause.** The JSON writer re-derived metrics by dividing running accumulators
(`bpp_sum / num_images`, etc.), but `eval.py` had already computed the final averaged
quantities (`Bit_rate`, `PSNR`, `MS_SSIM`, `encode_time`, `decode_time`). The result was
a second, incorrect averaging.

**Resolution.** Wrote the already-computed final values straight into the JSON.

```diff
- "bpp":  float(bpp_sum / num_images),
- "psnr": float(psnr_sum / num_images),
+ "bpp":  float(Bit_rate),
+ "psnr": float(PSNR),
```

---

## 4. Pretrained checkpoints could not be loaded (DDP `module.` prefix)

**Symptom.** Loading any of the authors' six pretrained MSE checkpoints raised a
state-dict key mismatch; training/evaluation could not start from them.

**Root cause.** The published checkpoints were produced with `DistributedDataParallel`,
which wraps the model and prefixes every parameter key with `module.`. Loading them into
a plain single-GPU model fails. Additionally, the checkpoints contain only `state_dict`
— not the optimizer, scheduler, or epoch counter our resume path assumed.

**Resolution.** Strip the `module.` prefix on load, and make optimizer/scheduler/epoch
restoration conditional on the keys actually being present.

```python
state_dict = {k.replace("module.", ""): v for k, v in checkpoint["state_dict"].items()}
net.load_state_dict(state_dict)

if args.continue_train:
    if "epoch"        in checkpoint: last_epoch = checkpoint["epoch"] + 1
    if "optimizer"    in checkpoint: optimizer.load_state_dict(checkpoint["optimizer"])
    if "aux_optimizer" in checkpoint: aux_optimizer.load_state_dict(checkpoint["aux_optimizer"])
    if "lr_scheduler" in checkpoint: lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
```

---

## 5. Entropy-model CDF buffers missing on load

**Symptom.** Even after fixing the `module.` prefix, `_update_registered_buffer` in
`models/dcae.py` raised a `KeyError` on the entropy model's cumulative-distribution
buffers (`_quantized_cdf`, `_cdf_length`, `_offset`).

**Root cause.** The original code had the prefix-handling logic commented out, so it
assumed every CDF buffer key was present under its bare name. Depending on how a
checkpoint was saved, these keys may carry the `module.` prefix or be absent entirely
(they are re-derived by `update()` before real entropy coding).

**Resolution.** Try the bare key, fall back to the `module.`-prefixed key, and return
gracefully if neither exists rather than crashing.

```python
if state_dict_key not in state_dict:
    state_dict_key = "module." + state_dict_key
if state_dict_key not in state_dict:
    return
```

---

## 6. ImageNet-1k streaming load crashed on current `datasets`

**Symptom.** Cell 5 of the Colab notebook aborted while building the ImageNet subset:

```
trust_remote_code is not supported anymore.
HfUriError: Invalid HF URI 'hf://datasets/imagenet-1k@...'.
Repository id must be 'namespace/name', got 'imagenet-1k'.
```

**Root causes.**
1. Current `datasets` / `huggingface_hub` releases have **removed `trust_remote_code`**,
   which the loader passed.
2. The dataset was requested by the bare id `imagenet-1k`; the new Hub URI parser
   requires a namespaced id (`namespace/name`).
3. A latent performance trap: the `train` split streams in **class order**, so the
   per-class cap (4 images × 1000 classes) would have forced iteration through nearly all
   1.28 M images — effectively hanging even once the crash was fixed.

**Resolution.** Load the namespaced repository, which now ships an auto-converted
**Parquet** export and therefore streams without any loading script; add a shuffle buffer
so a class-diverse subset is collected after reading on the order of 15–20 k images.

```python
ds = load_dataset("ILSVRC/imagenet-1k", split=split, streaming=True)
ds = ds.shuffle(seed=seed, buffer_size=20000)
```

---

## 7. Kodak set absent on a clean Colab clone

**Symptom.** Cell 8 (Kodak evaluation) failed with
`FileNotFoundError: [Errno 2] No such file or directory: 'datasets/kodak'`.

**Root cause.** `datasets/` is excluded by `.gitignore`, so the benchmark images are not
present in a fresh clone, and the notebook never invoked the existing downloader.

**Resolution.** Added an idempotent download cell to the notebook (before the evaluation
cells) that runs `scripts/download_kodak.py` into `datasets/kodak` only when the 24
images are not already present, restoring the directory layout `eval.py` expects.

---

## 8. Fine-tune ran zero epochs and saved no checkpoint

**Symptom.** Cell 11 (short fine-tune) printed `Fine-tuning complete.` and exited
cleanly, yet Cell 12 (re-evaluate fine-tuned model) aborted with:

```
FileNotFoundError: No fine-tuned checkpoint found in Drive — did Cell 11 complete?
```

The Cell 11 log loaded the model and the pretrained checkpoint, then jumped straight to
the notebook's own completion print — with **no `Train epoch …`, no `Test epoch …`**
lines and no traceback. Training never actually executed, so nothing was written to
Drive.

**Root causes.**
1. **The pretrained epoch counter zeroed out the training loop.** With
   `--continue_train` defaulting to `True`, `train.py` set
   `last_epoch = checkpoint["epoch"] + 1` from the *pretrained* checkpoint, then looped
   `for epoch in range(last_epoch, args.epochs)`. The pretrained checkpoint's epoch
   counter is far above the single fine-tune epoch, so the range was **empty** and
   `train_one_epoch` / `test_epoch` / `save_checkpoint` were all skipped. (The same flaw
   broke disconnect-resume: an interval checkpoint stores `epoch=0`, so a resume set
   `last_epoch=1` and again ran nothing.)
2. **Checkpoints were written to the wrong path.** `save_path` was built with
   `os.path.join(args.save_path, str(args.lmbda))` (no trailing slash) but
   `save_checkpoint` assembled filenames by string concatenation
   (`save_path + "checkpoint_latest.pth.tar"`), producing the mangled
   `.../finetune/0.0067checkpoint_latest.pth.tar`. The mid-epoch interval save also used
   `args.save_path` (no lambda subdir), so it missed the `0.0067/` directory entirely —
   exactly where both the notebook's resume probe and Cell 12 look.

**Resolution.** Decoupled the fine-tune from the pretrained epoch counter (always train
`args.epochs` fresh epochs from the loaded weights, while still restoring
optimizer/scheduler state when present), and switched every checkpoint write to
`os.path.join` against the lambda subdir so interval and end-of-epoch saves both land at
`.../finetune/<lambda>/checkpoint_{latest,best}.pth.tar`.

```python
# epoch loop no longer seeded from the pretrained checkpoint
if args.continue_train:
    if "optimizer" in checkpoint:     optimizer.load_state_dict(checkpoint["optimizer"])
    if "aux_optimizer" in checkpoint: aux_optimizer.load_state_dict(checkpoint["aux_optimizer"])
    if "lr_scheduler" in checkpoint:  lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
# ...
for epoch in range(last_epoch, args.epochs):   # last_epoch stays 0 -> one epoch runs

# checkpoints written into the lambda subdir with a real separator
torch.save(state, os.path.join(save_path, "checkpoint_latest.pth.tar"))
```

---

## 9. Stale Colab clone ran outdated `train.py`

**Symptom.** *Identical to Challenge #8, after #8 was already fixed and pushed.* Cell 11
printed `Fine-tuning complete.` and exited cleanly — with **no `Train epoch …`, no
`Test epoch …`** lines and no traceback — and Cell 12 again aborted with:

```
FileNotFoundError: No fine-tuned checkpoint found in Drive — did Cell 11 complete?
```

**Root cause.** The data and the code fix were both fine; the runtime simply wasn't running
the fixed code.

1. Cell 5 had built the full subset (**~4000** train images), so the dataloader was not empty.
2. The #8 fix (decoupling `last_epoch` from the pretrained checkpoint) was committed and
   **pushed** to the `imagenet-port` branch.
3. But Cell 3's logic was `if not os.path.exists(REPO_DIR): git clone … else: skip` — with
   **no `git pull`**. A Colab runtime carried over from an earlier session still had a
   `/content/DCAE` checkout from *before* the #8 push. Cell 3 saw it, printed
   `already exists — skipping`, and left it untouched. Cells 4–13 therefore executed the
   **stale `train.py`**, which still set `last_epoch = checkpoint["epoch"] + 1` → the epoch
   range was empty → training and saving were skipped, reproducing #8 exactly.

In short: pushing a fix is not the same as running it. The notebook never reconciled an
existing clone with the pushed branch, so the fix never reached the executing code.

**Resolution.** Made Cell 3 force-sync an existing clone to the latest pushed branch (a fresh
runtime still clones as before):

```diff
 if not os.path.exists(REPO_DIR):
     !git clone -b {GIT_BRANCH} {GITHUB_REPO} {REPO_DIR}
 else:
-    print(f"{REPO_DIR} already exists — skipping clone.")
+    print(f"{REPO_DIR} already exists — syncing to latest origin/{GIT_BRANCH}...")
+    !git -C {REPO_DIR} fetch origin {GIT_BRANCH}
+    !git -C {REPO_DIR} checkout {GIT_BRANCH}
+    !git -C {REPO_DIR} reset --hard origin/{GIT_BRANCH}
```

`reset --hard origin/{GIT_BRANCH}` guarantees the working tree matches the pushed branch even
if the cached checkout diverged or carried local edits.

As defence-in-depth against this whole class of *silent* "trains nothing" failures, `train.py`
now also prints the split sizes and raises immediately if the training split is empty, instead
of letting the epoch loop iterate zero batches:

```python
print(f"train images: {len(train_dataset)} | test images: {len(test_dataset)}")
if len(train_dataset) == 0:
    raise RuntimeError(f"No training images found under {args.dataset}/train — did the dataset build step (Cell 5) run?")
```

---

## 10. Sub-256 images crashed the fine-tune `RandomCrop` mid-epoch

**Symptom.** Cell 11 (fine-tune) started, loaded the checkpoint, then died inside the
dataloader with:

```
ValueError: Required crop size (256, 256) is larger than input image size (512, 242)
```

Because the run crashed before any checkpoint was written, Cell 12 then failed with
`FileNotFoundError: No fine-tuned checkpoint found in Drive — did Cell 11 complete?`.
The two errors looked unrelated but were the same failure: nothing trained, so nothing
was saved.

**Root cause.** `scripts/build_imagenet_subset.py` checked `min(W, H) >= min_side` (256)
**before** resizing, then shrank the long edge to `max_side` (512). Scaling the long edge
down scales the short edge with it, so an elongated image that passed the pre-resize check
(e.g. `1083x512`, short edge 512 ≥ 256) ended up at `512x242` — short edge **below** 256.
That sub-256 image entered the subset, and `train.py`'s `transforms.RandomCrop(256)` (no
padding) raised `ValueError` the moment the dataloader reached it.

**Resolution.** Two layers of defence:

1. **Build side** — re-check the size *after* resizing and skip anything still below
   `min_side`, so no sub-256 image reaches the subset:

```python
img = resize_long_edge(img, args.max_side)
# resizing the long edge scales the short edge down with it
if not passes_size(img, args.min_side):
    continue
```

2. **Training side** — make the crop tolerant so a stray small image pads instead of
   killing the whole run (and so an already-built subset trains without a costly rebuild):

```python
transforms.RandomCrop(args.patch_size, pad_if_needed=True)
```

`CenterCrop` (test transform) already zero-pads undersized inputs, so only the train
`RandomCrop` needed the guard.

---

## Lessons learned

- **Distributed-training artefacts leak into single-GPU use.** The `module.` prefix
  (challenges 4–5) is the most common friction point when reusing published checkpoints;
  prefix-stripping and tolerant buffer loading should be the default, not an afterthought.
- **Mixed precision is not uniform across a model.** The rate-distortion auxiliary loss
  must stay in fp32 while the main reconstruction loss runs in fp16 (challenge 2).
- **Pin to library behaviour, not just versions.** The Hugging Face `trust_remote_code`
  removal (challenge 6) broke working code with no version change on our side; preferring
  the Parquet export future-proofs the data pipeline.
- **A clean clone must be self-sufficient.** Anything excluded by `.gitignore`
  (datasets, checkpoints) needs an explicit, idempotent fetch step in the runbook
  (challenges 6–7).
- **Fine-tuning is not resuming.** A pretrained checkpoint's epoch counter must not drive
  the fine-tune loop bound, or the run silently trains nothing (challenge 8). Build save
  paths with `os.path.join`, never string concatenation — a missing separator hides the
  output where the next stage can't find it.
- **Pushing a fix is not running it.** A notebook that conditionally skips cloning must still
  `git pull` / `reset --hard` to the pushed branch, or a reused runtime keeps executing an old
  checkout and a bug "reappears" after it was already fixed (challenge 9). Pair this with loud
  guards (e.g. asserting a non-empty dataset) so silent no-ops surface as errors.
- **Filter the data the way the model consumes it.** A size check must run on the final image,
  not the original — resizing the long edge shrinks the short edge with it, so a pre-resize
  filter let sub-256 images through and crashed the crop (challenge 10). Combine an upstream
  filter with a tolerant transform (`pad_if_needed=True`) so one bad sample degrades instead of
  aborting the run.
