"""
Compute Bjøntegaard-Delta (BD) Rate and BD-PSNR between two RD curves.

BD-Rate  < 0  means "ours achieves the same quality at fewer bits".
BD-PSNR  > 0  means "ours achieves higher quality at the same bitrate".

Usage — compare your results.json files vs VTM anchor from RD_data.json:
    python scripts/bd_rate.py \
        --ours results/kodak_mse_0.0018.json results/kodak_mse_0.0035.json \
               results/kodak_mse_0.0067.json results/kodak_mse_0.0130.json \
               results/kodak_mse_0.0250.json results/kodak_mse_0.0500.json \
        --anchor_json RD_data.json \
        --anchor_key vtm_kodak \
        --metric psnr \
        --out reports/bdrate_kodak.csv

Or compare two JSON-list files directly:
    python scripts/bd_rate.py --ours_json ours.json --ref_json vtm.json
    where each JSON is a list of {"bpp": float, "psnr": float, ...}

Reference:
    Bjøntegaard (2001), "Calculation of average PSNR differences between RD-curves",
    ITU-T SG16 VCEG-M33.
"""
import argparse
import json
import math
import sys
from pathlib import Path


# ──────────────────────────────────────────────────────────────
# core algorithm  (Piecewise-cubic polynomial integration)
# ──────────────────────────────────────────────────────────────

def _poly_area(bpp_pts, quality_pts):
    """Integrate quality over log(bpp) using cubic polynomial fit."""
    import numpy as np
    log_bpp = [math.log(b) for b in bpp_pts]
    coeffs = np.polyfit(log_bpp, quality_pts, deg=3)
    poly = np.poly1d(coeffs)
    lo, hi = min(log_bpp), max(log_bpp)
    integral = np.polyint(poly)
    return float(integral(hi) - integral(lo)), lo, hi


def bd_rate(ref_bpp, ref_quality, test_bpp, test_quality):
    """
    Compute BD-Rate (%) between reference and test RD curves.
    Negative = test is better (fewer bits for same quality).
    Each input is a list of floats, sorted by bpp.
    """
    import numpy as np

    ref_bpp = list(ref_bpp)
    ref_quality = list(ref_quality)
    test_bpp = list(test_bpp)
    test_quality = list(test_quality)

    # sort both by bpp
    ref_pts = sorted(zip(ref_bpp, ref_quality))
    test_pts = sorted(zip(test_bpp, test_quality))
    rb, rq = zip(*ref_pts)
    tb, tq = zip(*test_pts)

    # overlap region in log-bpp space
    rb_log = [math.log(b) for b in rb]
    tb_log = [math.log(b) for b in tb]
    lo = max(min(rb_log), min(tb_log))
    hi = min(max(rb_log), max(tb_log))

    if lo >= hi:
        raise ValueError("RD curves do not overlap in bpp range — cannot compute BD-Rate.")

    ref_coeffs = np.polyfit(rq, rb_log, deg=3)
    test_coeffs = np.polyfit(tq, tb_log, deg=3)

    q_lo = max(min(rq), min(tq))
    q_hi = min(max(rq), max(tq))

    ref_int = np.polyint(np.poly1d(ref_coeffs))
    test_int = np.polyint(np.poly1d(test_coeffs))

    ref_area = float(ref_int(q_hi) - ref_int(q_lo))
    test_area = float(test_int(q_hi) - test_int(q_lo))

    avg_exp = (test_area - ref_area) / (q_hi - q_lo)
    return (math.exp(avg_exp) - 1.0) * 100.0


def bd_psnr(ref_bpp, ref_quality, test_bpp, test_quality):
    """
    Compute BD-PSNR (dB) — positive means test has higher quality at same bitrate.
    """
    import numpy as np

    ref_pts = sorted(zip(ref_bpp, ref_quality))
    test_pts = sorted(zip(test_bpp, test_quality))
    rb, rq = zip(*ref_pts)
    tb, tq = zip(*test_pts)

    rb_log = [math.log(b) for b in rb]
    tb_log = [math.log(b) for b in tb]
    lo = max(min(rb_log), min(tb_log))
    hi = min(max(rb_log), max(tb_log))

    if lo >= hi:
        raise ValueError("RD curves do not overlap.")

    ref_coeffs = np.polyfit(rb_log, rq, deg=3)
    test_coeffs = np.polyfit(tb_log, tq, deg=3)

    ref_int = np.polyint(np.poly1d(ref_coeffs))
    test_int = np.polyint(np.poly1d(test_coeffs))

    ref_area = float(ref_int(hi) - ref_int(lo))
    test_area = float(test_int(hi) - test_int(lo))

    return (test_area - ref_area) / (hi - lo)


# ──────────────────────────────────────────────────────────────
# loading helpers
# ──────────────────────────────────────────────────────────────

def load_points_from_json_files(paths, metric="psnr"):
    """
    Load individual result JSON files (one per checkpoint) with keys:
        {"bpp": float, "psnr": float, "msssim": float, ...}
    """
    bpps, quals = [], []
    for p in paths:
        with open(p) as f:
            d = json.load(f)
        bpps.append(float(d["bpp"]))
        quals.append(float(d[metric]))
    pts = sorted(zip(bpps, quals))
    return [b for b, _ in pts], [q for _, q in pts]


def load_points_from_list_json(path, bpp_key="bpp", metric="psnr"):
    """Load from a JSON file that is a list of dicts."""
    with open(path) as f:
        data = json.load(f)
    pts = sorted((float(d[bpp_key]), float(d[metric])) for d in data)
    return [b for b, _ in pts], [q for _, q in pts]


def load_anchor_from_rd_data(rd_json_path, anchor_key, metric="psnr"):
    """
    Load from RD_data.json (the file shipped with DCAE).
    Expected structure: {anchor_key: [[bpp, psnr, msssim], ...]}
    """
    with open(rd_json_path) as f:
        data = json.load(f)
    if anchor_key not in data:
        available = list(data.keys())
        raise KeyError(f"Key '{anchor_key}' not in {rd_json_path}. Available: {available}")
    rows = data[anchor_key]
    metric_idx = {"psnr": 1, "msssim": 2}.get(metric, 1)
    pts = sorted((float(r[0]), float(r[metric_idx])) for r in rows)
    return [b for b, _ in pts], [q for _, q in pts]


# ──────────────────────────────────────────────────────────────
# main
# ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Compute BD-Rate / BD-PSNR")

    # individual result files (one per checkpoint)
    parser.add_argument("--ours", nargs="*", default=None,
                        help="Per-checkpoint result JSON files (eval.py --results_json output)")

    # or a pre-aggregated list JSON
    parser.add_argument("--ours_json", default=None,
                        help="Single JSON list with all operating points (alternative to --ours)")

    # anchor: RD_data.json from the DCAE repo
    parser.add_argument("--anchor_json", default="RD_data.json",
                        help="Path to RD_data.json (shipped with the DCAE repo)")
    parser.add_argument("--anchor_key", default="vtm_kodak",
                        help="Key inside RD_data.json to use as anchor (default: vtm_kodak)")

    # or supply a plain list JSON as anchor
    parser.add_argument("--ref_json", default=None,
                        help="Alternative anchor as a plain JSON list")

    parser.add_argument("--metric", choices=["psnr", "msssim"], default="psnr")
    parser.add_argument("--out", default=None, help="Write CSV results to this path")
    args = parser.parse_args()

    # load test curve
    if args.ours:
        test_bpp, test_qual = load_points_from_json_files(args.ours, args.metric)
    elif args.ours_json:
        test_bpp, test_qual = load_points_from_list_json(args.ours_json, metric=args.metric)
    else:
        sys.exit("Provide --ours <files...> or --ours_json <file>")

    # load reference curve
    if args.ref_json:
        ref_bpp, ref_qual = load_points_from_list_json(args.ref_json, metric=args.metric)
    else:
        ref_bpp, ref_qual = load_anchor_from_rd_data(args.anchor_json, args.anchor_key, args.metric)

    bdr = bd_rate(ref_bpp, ref_qual, test_bpp, test_qual)
    bdp = bd_psnr(ref_bpp, ref_qual, test_bpp, test_qual)

    print(f"\nBD-Rate  (vs {args.anchor_key}):  {bdr:+.2f} %   (negative = better)")
    print(f"BD-PSNR  (vs {args.anchor_key}):  {bdp:+.3f} dB  (positive = better)")
    print(f"Metric: {args.metric},  {len(test_bpp)} test points,  {len(ref_bpp)} ref points")

    if args.out:
        import csv
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["anchor", "metric", "bd_rate_pct", "bd_psnr_db", "n_test_pts", "n_ref_pts"])
            w.writerow([args.anchor_key, args.metric, f"{bdr:.4f}", f"{bdp:.4f}",
                        len(test_bpp), len(ref_bpp)])
        print(f"Results written to {args.out}")


if __name__ == "__main__":
    main()
