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
