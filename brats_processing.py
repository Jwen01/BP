"""Convert raw BraTS 2019 training cases to BP-AdMT/SSL4MIS HDF5 files.

The benchmark uses FLAIR as a single input channel and merges every non-zero
BraTS label into one whole-tumour foreground class. The official SSL4MIS split
files (250 train / 25 validation / 60 test) must exist in the output directory.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import h5py
import nibabel as nib
import numpy as np
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RAW_ROOT = (
    PROJECT_ROOT / "data" / "BraTS2019_raw" / "MICCAI_BraTS_2019_Data_Training"
)
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "data" / "BraTS2019"
EXPECTED_SPLIT_SIZES = {"train.txt": 250, "val.txt": 25, "test.txt": 60}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing HDF5 cases instead of validating and skipping them.",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Validate split files and existing HDF5 files without converting data.",
    )
    return parser.parse_args()


def read_split_files(output_root: Path) -> dict[str, list[str]]:
    splits: dict[str, list[str]] = {}
    for filename, expected_size in EXPECTED_SPLIT_SIZES.items():
        path = output_root / filename
        if not path.is_file():
            raise FileNotFoundError(f"Missing benchmark split file: {path}")
        names = [
            line.strip().split(",")[0]
            for line in path.read_text().splitlines()
            if line.strip()
        ]
        if len(names) != expected_size:
            raise ValueError(
                f"{path} contains {len(names)} cases; expected {expected_size}."
            )
        if len(names) != len(set(names)):
            raise ValueError(f"Duplicate case names found in {path}.")
        splits[filename] = names

    all_names = [name for names in splits.values() for name in names]
    if len(all_names) != len(set(all_names)):
        raise ValueError("The train/validation/test split files overlap.")
    return splits


def discover_cases(raw_root: Path) -> dict[str, Path]:
    cases: dict[str, Path] = {}
    for grade in ("HGG", "LGG"):
        grade_root = raw_root / grade
        if not grade_root.is_dir():
            raise FileNotFoundError(f"Missing raw BraTS directory: {grade_root}")
        for case_dir in sorted(grade_root.glob("BraTS19*")):
            if case_dir.is_dir():
                if case_dir.name in cases:
                    raise ValueError(f"Duplicate raw case name: {case_dir.name}")
                cases[case_dir.name] = case_dir
    return cases


def modality_path(case_dir: Path, suffix: str) -> Path:
    stem = case_dir / f"{case_dir.name}_{suffix}"
    for extension in (".nii", ".nii.gz"):
        candidate = Path(f"{stem}{extension}")
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Missing {suffix} volume for {case_dir.name}")


def normalize_volume(image: np.ndarray) -> np.ndarray:
    image = image.astype(np.float32, copy=False)
    std = float(image.std())
    if not np.isfinite(std) or std <= 0:
        raise ValueError("Cannot normalize a volume with zero or invalid standard deviation.")
    return ((image - float(image.mean())) / std).astype(np.float32, copy=False)


def validate_h5(
    path: Path, expected_shape: tuple[int, ...] | None = None
) -> tuple[int, ...]:
    with h5py.File(path, "r") as handle:
        if "image" not in handle or "label" not in handle:
            raise ValueError(f"{path} does not contain both 'image' and 'label'.")
        image = handle["image"]
        label = handle["label"]
        if image.shape != label.shape or image.ndim != 3:
            raise ValueError(
                f"Invalid image/label shapes in {path}: {image.shape} and {label.shape}."
            )
        if expected_shape is not None and image.shape != expected_shape:
            raise ValueError(
                f"Unexpected shape in {path}: {image.shape}; expected {expected_shape}."
            )
        label_values = np.unique(label[:])
        if not set(label_values.tolist()).issubset({0, 1}):
            raise ValueError(f"Non-binary labels in {path}: {label_values.tolist()}")
        if image.dtype != np.float32 or label.dtype != np.uint8:
            raise ValueError(
                f"Unexpected dtypes in {path}: image={image.dtype}, label={label.dtype}."
            )
        return tuple(image.shape)


def convert_case(case_dir: Path, output_path: Path, overwrite: bool) -> None:
    if output_path.exists() and not overwrite:
        validate_h5(output_path)
        return

    flair = nib.load(modality_path(case_dir, "flair")).get_fdata(dtype=np.float32)
    segmentation = nib.load(modality_path(case_dir, "seg")).get_fdata(dtype=np.float32)
    if flair.shape != segmentation.shape or flair.ndim != 3:
        raise ValueError(
            f"Shape mismatch for {case_dir.name}: flair={flair.shape}, "
            f"seg={segmentation.shape}."
        )

    image = normalize_volume(flair)
    label = (segmentation > 0).astype(np.uint8)
    temporary_path = output_path.with_suffix(".h5.tmp")
    try:
        with h5py.File(temporary_path, "w") as handle:
            handle.create_dataset(
                "image", data=image, compression="gzip", compression_opts=4, chunks=True
            )
            handle.create_dataset(
                "label", data=label, compression="gzip", compression_opts=4, chunks=True
            )
            handle.attrs["case_name"] = case_dir.name
            handle.attrs["tumor_type"] = case_dir.parent.name
            handle.attrs["modality"] = "flair"
            handle.attrs["target"] = "whole_tumor"
        validate_h5(temporary_path, expected_shape=tuple(flair.shape))
        os.replace(temporary_path, output_path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def main() -> None:
    args = parse_args()
    raw_root = args.raw_root.resolve()
    output_root = args.output_root.resolve()
    h5_root = output_root / "BraTSh5"
    output_root.mkdir(parents=True, exist_ok=True)

    splits = read_split_files(output_root)
    expected_names = {name for names in splits.values() for name in names}
    cases = discover_cases(raw_root)
    raw_names = set(cases)
    if raw_names != expected_names:
        missing = sorted(expected_names - raw_names)
        extra = sorted(raw_names - expected_names)
        raise ValueError(
            "Raw cases do not match the benchmark splits. "
            f"Missing={missing[:10]} (total {len(missing)}), "
            f"extra={extra[:10]} (total {len(extra)})."
        )

    h5_root.mkdir(parents=True, exist_ok=True)
    if not args.verify_only:
        ordered_names = [
            name
            for split in ("train.txt", "val.txt", "test.txt")
            for name in splits[split]
        ]
        for name in tqdm(ordered_names, desc="Converting BraTS2019"):
            convert_case(cases[name], h5_root / f"{name}.h5", args.overwrite)

    output_files = sorted(h5_root.glob("*.h5"))
    output_names = {path.stem for path in output_files}
    if output_names != expected_names:
        missing = sorted(expected_names - output_names)
        extra = sorted(output_names - expected_names)
        raise ValueError(
            f"HDF5 set mismatch: missing={missing[:10]} (total {len(missing)}), "
            f"extra={extra[:10]} (total {len(extra)})."
        )

    shapes: dict[tuple[int, ...], int] = {}
    for path in tqdm(output_files, desc="Verifying BraTS2019"):
        shape = validate_h5(path)
        shapes[shape] = shapes.get(shape, 0) + 1
    print(
        "BraTS2019 ready: "
        f"train={len(splits['train.txt'])}, val={len(splits['val.txt'])}, "
        f"test={len(splits['test.txt'])}, shapes={shapes}, output={output_root}"
    )


if __name__ == "__main__":
    main()
