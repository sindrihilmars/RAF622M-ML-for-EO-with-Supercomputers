#!/usr/bin/env python3
"""
Extract training patches from aligned Sentinel-2 data using Level-1 CORINE labels.

Expects an aligned data directory produced by lab4_revisit (or equivalent),
containing:
  *_stacked.tif       - stacked S2 bands (B02, B03, B04, B08)
  labels_*.tif        - 5-class Level-1 label rasters (values 1-5)

Writes per-tile patches_*_data.npz + patches_*_metadata.json, then a combined
combined_training_data.npz to the output directory.

Usage:
    python extract_patches_level1.py <aligned_data_dir> <output_dir> <patch_size>
                                     [--stride N] [--max-patches N]
                                     [--normalization {percentile,minmax,zscore,none}]
                                     [--no-skip-existing]
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import rasterio

# ---------------------------------------------------------------------------
# Label definitions
# ---------------------------------------------------------------------------

LEVEL1_CLASSES = {
    1: "Artificial surfaces",
    2: "Agricultural areas",
    3: "Forest and semi-natural areas",
    4: "Wetlands",
    5: "Water bodies",
}

# ---------------------------------------------------------------------------
# Normalization helpers
# ---------------------------------------------------------------------------

def normalize_minmax(data, min_val=0, max_val=10000):
    data = data.astype(np.float32)
    return np.clip((data - min_val) / (max_val - min_val + 1e-6), 0, 1)


def normalize_zscore(data, mean=None, std=None):
    data = data.astype(np.float32)
    if mean is None:
        mean = np.mean(data)
    if std is None:
        std = np.std(data)
    return (data - mean) / (std + 1e-6), float(mean), float(std)


def normalize_percentile(data, low_pct=2, high_pct=98):
    data = data.astype(np.float32)
    low_val = np.percentile(data, low_pct)
    high_val = np.percentile(data, high_pct)
    return np.clip((data - low_val) / (high_val - low_val + 1e-6), 0, 1), float(low_val), float(high_val)


def normalize_data(data, method="percentile", **kwargs):
    if method == "minmax":
        normalized = normalize_minmax(data, **kwargs)
        params = {"method": "minmax", **kwargs}
    elif method == "zscore":
        normalized, mean, std = normalize_zscore(data, **kwargs)
        params = {"method": "zscore", "mean": mean, "std": std}
    elif method == "percentile":
        low_pct = kwargs.get("low_pct", 2)
        high_pct = kwargs.get("high_pct", 98)
        normalized, low_val, high_val = normalize_percentile(data, low_pct, high_pct)
        params = {"method": "percentile", "low_pct": low_pct, "high_pct": high_pct,
                  "low_val": low_val, "high_val": high_val}
    else:
        normalized = data.astype(np.float32)
        params = {"method": "none"}
    return normalized, params

# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

def find_aligned_pairs(data_dir):
    """Return list of dicts with s2_path, labels_path, tile_name."""
    data_path = Path(data_dir)
    if not data_path.exists():
        print(f"ERROR: Aligned data directory not found: {data_dir}")
        return []

    pairs = []
    for s2_path in sorted(data_path.glob("*_stacked.tif")):
        tile_name = s2_path.stem.replace("_stacked", "")
        labels_path = data_path / f"labels_{tile_name}.tif"
        if labels_path.exists():
            pairs.append({"s2_path": s2_path, "labels_path": labels_path, "tile_name": tile_name})
        else:
            print(f"  WARNING: No labels file for {tile_name}, skipping.")
    return pairs

# ---------------------------------------------------------------------------
# Patch extraction
# ---------------------------------------------------------------------------

def extract_patches(s2_path, labels_path, patch_size, stride, max_patches,
                    normalization, norm_kwargs):
    """
    Extract patches with Level-1 labels (1-5) via majority vote.

    Returns (patches, labels, metadata) or (None, None, None) on failure.
    """
    try:
        with rasterio.open(s2_path) as s2_src, rasterio.open(labels_path) as lbl_src:
            height, width = s2_src.height, s2_src.width
            n_bands = s2_src.count
            s2_data = s2_src.read()        # (bands, H, W)
            labels_data = lbl_src.read(1)  # (H, W)

        if stride is None:
            stride = patch_size

        print(f"    Image size: {width} x {height} px | bands: {n_bands} | "
              f"patch: {patch_size} stride: {stride}")

        print(f"    Applying {normalization} normalization...")
        s2_norm, norm_params = normalize_data(s2_data, normalization, **norm_kwargs)

        patches, patch_labels = [], []

        for y in range(0, height - patch_size + 1, stride):
            for x in range(0, width - patch_size + 1, stride):
                # Skip patches containing zero-valued S2 pixels (missing data)
                orig_patch = s2_data[:, y:y + patch_size, x:x + patch_size]
                if np.any(orig_patch == 0):
                    continue

                # Majority-vote label from valid Level-1 pixels (1-5)
                lbl_patch = labels_data[y:y + patch_size, x:x + patch_size]
                valid_mask = (lbl_patch >= 1) & (lbl_patch <= 5)
                valid_lbls = lbl_patch[valid_mask]

                if valid_lbls.size == 0:
                    continue
                # Require at least half the patch to have valid labels
                if valid_lbls.size < (patch_size * patch_size) // 2:
                    continue

                counts = np.bincount(valid_lbls.astype(np.int64), minlength=6)
                label = int(counts.argmax())
                if label < 1 or label > 5:
                    continue

                patch = np.transpose(s2_norm[:, y:y + patch_size, x:x + patch_size], (1, 2, 0))
                patches.append(patch)
                patch_labels.append(label)

                if len(patches) >= max_patches:
                    break

            if len(patches) >= max_patches:
                break

            if y % 2000 == 0 and y > 0:
                print(f"    Progress: {len(patches):,} patches...")

        if not patches:
            return None, None, None

        patches = np.array(patches, dtype=np.float32)
        patch_labels = np.array(patch_labels, dtype=np.uint8)

        unique_lbls, counts = np.unique(patch_labels, return_counts=True)
        metadata = {
            "s2_file": Path(s2_path).name,
            "labels_file": Path(labels_path).name,
            "patch_size": patch_size,
            "stride": stride,
            "n_patches": len(patches),
            "n_bands": int(patches.shape[3]),
            "n_classes": int(len(unique_lbls)),
            "label_type": "level1_corine",
            "label_distribution": {int(k): int(v) for k, v in zip(unique_lbls, counts)},
            "extraction_date": datetime.now().isoformat(),
            "patch_shape": list(patches.shape),
            "bands": ["B02", "B03", "B04", "B08"],
            "normalization": norm_params,
        }
        return patches, patch_labels, metadata

    except Exception as exc:
        import traceback
        print(f"  ERROR: Patch extraction failed: {exc}")
        traceback.print_exc()
        return None, None, None

# ---------------------------------------------------------------------------
# Per-tile processing
# ---------------------------------------------------------------------------

def process_tile(pair, output_dir, patch_size, stride, max_patches,
                 normalization, norm_kwargs, skip_existing):
    tile_name = pair["tile_name"]
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    npz_path = output_dir / f"patches_{tile_name}_data.npz"
    meta_path = output_dir / f"patches_{tile_name}_metadata.json"

    result = {"tile": tile_name, "status": "unknown", "n_patches": 0,
              "n_classes": 0, "output_file": None}

    if skip_existing and npz_path.exists():
        print("  Skipping (output exists).")
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
            result["n_patches"] = meta.get("n_patches", 0)
            result["n_classes"] = meta.get("n_classes", 0)
        result["status"] = "skipped"
        result["output_file"] = str(npz_path)
        return result

    print("  Extracting patches...")
    patches, labels, metadata = extract_patches(
        pair["s2_path"], pair["labels_path"],
        patch_size, stride, max_patches, normalization, norm_kwargs,
    )

    if patches is None:
        result["status"] = "no_patches"
        return result

    print(f"  Saving {len(patches):,} patches...")
    np.savez_compressed(npz_path, patches=patches, labels=labels)
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    result.update({"status": "success", "n_patches": len(patches),
                   "n_classes": metadata["n_classes"], "output_file": str(npz_path)})
    return result

# ---------------------------------------------------------------------------
# Combine tiles
# ---------------------------------------------------------------------------

def combine_datasets(results, output_path):
    all_patches, all_labels = [], []

    for r in results:
        if r["status"] not in ("success", "skipped"):
            continue
        if not r["output_file"] or not Path(r["output_file"]).exists():
            continue
        data = np.load(r["output_file"])
        all_patches.append(data["patches"])
        all_labels.append(data["labels"])
        print(f"  Loaded {len(data['labels']):,} patches from {r['tile'][:50]}...")

    if not all_patches:
        print("Nothing to combine.")
        return

    combined_patches = np.concatenate(all_patches, axis=0)
    combined_labels = np.concatenate(all_labels, axis=0)

    idx = np.random.permutation(len(combined_labels))
    combined_patches = combined_patches[idx]
    combined_labels = combined_labels[idx]

    np.savez_compressed(output_path, patches=combined_patches, labels=combined_labels)

    unique, counts = np.unique(combined_labels, return_counts=True)
    print(f"\nCombined dataset saved: {output_path}")
    print(f"  Total patches : {len(combined_labels):,}")
    print(f"  Unique classes: {len(unique)}")
    print(f"  File size     : {Path(output_path).stat().st_size / 1024**2:.1f} MB")
    print("\nLevel-1 class distribution:")
    for lbl, cnt in zip(unique, counts):
        name = LEVEL1_CLASSES.get(int(lbl), "Unknown")
        print(f"  Class {lbl}: {name:<35s} {cnt:>8,}  ({cnt/len(combined_labels)*100:5.1f}%)")






def combine_datasets_early_fusion(results, output_path, max_samples=5000,
                                   augment=True):
    """
    Combine per-tile patch files by stacking temporal acquisitions channel-wise.
    Optionally augments with 90/180/270° rotations to reach max_samples total.

    Each result should be a different acquisition date of the same spatial tile,
    so that patch index i corresponds to the same ground location across all
    files.  Patches are extracted in raster-scan order with the same
    stride/max_patches, which guarantees positional alignment as long as all
    tiles were processed with identical parameters.

    Input shape per file : (N, H, W, B)    e.g. (50000, 32, 32, 4)
    Output shape         : (N_out, H, W, B*T)

    Labels are taken from the first file (land cover assumed stable across
    dates).

    Parameters
    ----------
    results : list of dicts
        Result dicts from process_tile(), each with an 'output_file' key.
    output_path : Path or str
        Where to save the fused .npz file.
    max_samples : int
        Only used when augment=True. Target total number of patches.
        If the fused dataset is smaller, rotated copies (90°, 180°, 270°)
        are added until max_samples is reached. If larger, it is randomly
        subsampled down to max_samples.
    augment : bool
        If True (default), augment/subsample to max_samples after fusion.
        If False, save all fused patches without any rotation augmentation.
        Use False when augmentation will be applied after a train/val/test
        split to avoid data leakage.
    """
    patches_list, labels_list = [], []

    for r in results:
        if r["status"] not in ("success", "skipped"):
            continue
        if not r["output_file"] or not Path(r["output_file"]).exists():
            continue
        data = np.load(r["output_file"])
        patches_list.append(data["patches"])
        labels_list.append(data["labels"])
        print(f"  Loaded {data['patches'].shape} from {r['tile'][:50]}...")

    if not patches_list:
        print("Nothing to fuse.")
        return None, None

    # Truncate to the smallest tile if counts differ
    n_patches = [p.shape[0] for p in patches_list]
    if len(set(n_patches)) > 1:
        print(f"  WARNING: patch counts differ {n_patches} "
              f"— truncating to {min(n_patches):,}")
        min_n = min(n_patches)
        patches_list = [p[:min_n] for p in patches_list]
        labels_list  = [l[:min_n] for l in labels_list]

    # Stack along new time axis then flatten bands × time → channels
    # stacked: (N, H, W, B, T)
    stacked = np.stack(patches_list, axis=-1)
    N, H, W, B, T = stacked.shape
    fused_patches = stacked.reshape(N, H, W, B * T)   # (N, H, W, B*T)
    fused_labels  = labels_list[0]

    if augment:
        # --- Augment or subsample to reach max_samples ---
        if N < max_samples:
            needed = max_samples - N
            print(f"  Augmenting: {N:,} → {max_samples:,} "
                  f"({needed:,} rotated copies needed)")

            aug_patches, aug_labels = [], []
            rng = np.random.default_rng()

            while len(aug_patches) < needed:
                batch   = min(needed - len(aug_patches), N)
                indices = rng.integers(0, N, size=batch)
                ks      = rng.integers(1, 4, size=batch)  # 90°, 180°, 270°
                for i, k in zip(indices, ks):
                    rotated = np.rot90(fused_patches[i], k=k, axes=(0, 1))
                    aug_patches.append(rotated)
                    aug_labels.append(fused_labels[i])

            fused_patches = np.concatenate(
                [fused_patches, np.array(aug_patches, dtype=np.float32)],
                axis=0,
            )
            fused_labels = np.concatenate(
                [fused_labels, np.array(aug_labels, dtype=np.uint8)], axis=0
            )
        elif N > max_samples:
            print(f"  Subsampling: {N:,} → {max_samples:,}")
            idx = np.random.choice(N, max_samples, replace=False)
            fused_patches = fused_patches[idx]
            fused_labels  = fused_labels[idx]
    else:
        print(f"  augment=False — keeping all {N:,} fused patches "
              f"(augmentation deferred to post-split)")

    # Shuffle
    idx = np.random.permutation(len(fused_labels))
    fused_patches = fused_patches[idx]
    fused_labels  = fused_labels[idx]

    np.savez_compressed(output_path, patches=fused_patches, labels=fused_labels)

    unique, counts = np.unique(fused_labels, return_counts=True)
    print(f"\nEarly-fusion dataset saved: {output_path}")
    print(f"  Dates fused     : {T}")
    print(f"  Output shape    : {fused_patches.shape}  (N, H, W, bands×dates)")
    print(f"  Unique classes  : {len(unique)}")
    print(f"  File size       : "
          f"{Path(output_path).stat().st_size / 1024**2:.1f} MB")
    print("\nLevel-1 class distribution:")
    for lbl, cnt in zip(unique, counts):
        name = LEVEL1_CLASSES.get(int(lbl), "Unknown")
        print(f"  Class {lbl}: {name:<35s} {cnt:>8,}  "
              f"({cnt/len(fused_labels)*100:5.1f}%)")

    return fused_patches, fused_labels

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Extract patches from aligned S2 data using Level-1 CORINE labels.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("aligned_data_dir", help="Directory with *_stacked.tif and labels_*.tif files")
    p.add_argument("output_dir", help="Where to write patches_*.npz and combined_training_data.npz")
    p.add_argument("patch_size", type=int, help="Patch size in pixels (e.g. 3 = 30m x 30m)")
    p.add_argument("--stride", type=int, default=None,
                   help="Extraction stride (default: same as patch_size, no overlap)")
    p.add_argument("--max-patches", type=int, default=50000,
                   help="Maximum patches per tile")
    p.add_argument("--normalization", choices=["percentile", "minmax", "zscore", "none"],
                   default="percentile")
    p.add_argument("--percentile-low", type=float, default=2)
    p.add_argument("--percentile-high", type=float, default=98)
    p.add_argument("--no-skip-existing", action="store_true",
                   help="Re-process tiles even if output already exists")
    return p.parse_args()


def main():
    args = parse_args()

    norm_kwargs = {}
    if args.normalization == "percentile":
        norm_kwargs = {"low_pct": args.percentile_low, "high_pct": args.percentile_high}

    print("=" * 70)
    print("EXTRACTING TRAINING PATCHES (Level-1 CORINE labels)")
    print("=" * 70)
    print(f"Start time       : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Aligned data dir : {args.aligned_data_dir}")
    print(f"Output dir       : {args.output_dir}")
    print(f"Patch size       : {args.patch_size}x{args.patch_size} ({args.patch_size * 10}m)")
    print(f"Max patches/tile : {args.max_patches:,}")
    print(f"Normalization    : {args.normalization}")
    print()

    pairs = find_aligned_pairs(args.aligned_data_dir)
    if not pairs:
        print("No aligned pairs found. Exiting.")
        sys.exit(1)

    print(f"Found {len(pairs)} aligned tile pair(s):")
    for i, p in enumerate(pairs, 1):
        print(f"  {i}. {p['tile_name'][:70]}")
    print()

    all_results = []
    for i, pair in enumerate(pairs, 1):
        print(f"[{i}/{len(pairs)}] {pair['tile_name'][:60]}...")
        result = process_tile(
            pair=pair,
            output_dir=args.output_dir,
            patch_size=args.patch_size,
            stride=args.stride,
            max_patches=args.max_patches,
            normalization=args.normalization,
            norm_kwargs=norm_kwargs,
            skip_existing=not args.no_skip_existing,
        )
        all_results.append(result)

        if result["status"] == "success":
            print(f"  OK: {result['n_patches']:,} patches, {result['n_classes']} classes")
        elif result["status"] == "skipped":
            print(f"  Skipped: {result['n_patches']:,} patches already on disk")
        else:
            print(f"  FAILED: {result['status']}")

    # Summary
    successful = [r for r in all_results if r["status"] == "success"]
    skipped    = [r for r in all_results if r["status"] == "skipped"]
    failed     = [r for r in all_results if r["status"] not in ("success", "skipped")]

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"End time   : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Successful : {len(successful)}")
    print(f"Skipped    : {len(skipped)}")
    print(f"Failed     : {len(failed)}")
    total = sum(r["n_patches"] for r in all_results if r["n_patches"])
    print(f"Total patches extracted: {total:,}")

    # Combine
    valid = [r for r in all_results if r["status"] in ("success", "skipped") and r["output_file"]]
    if len(valid) > 1:
        print("\nCombining tile datasets...")
        combined_path = Path(args.output_dir) / "combined_training_data.npz"
        combine_datasets(valid, combined_path)
    elif len(valid) == 1:
        print(f"\nOnly one tile — training data at: {valid[0]['output_file']}")


if __name__ == "__main__":
    main()