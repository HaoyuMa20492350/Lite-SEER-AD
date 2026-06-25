from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageFile
from torch.utils.data import Dataset
from torchvision import transforms

from seer_ad_v2.utils.image import list_images, load_rgb

ImageFile.LOAD_TRUNCATED_IMAGES = True

MVTEC_AD2_CATEGORIES = [
    "can",
    "fabric",
    "fruit_jelly",
    "rice",
    "sheet_metal",
    "vial",
    "wallplugs",
    "walnuts",
]


@dataclass(frozen=True)
class ImageRecord:
    image_path: Path
    label: int
    mask_path: Path | None
    category: str
    defect_type: str


def _mask_candidates(mask_dir: Path, stem: str) -> list[Path]:
    return [
        mask_dir / f"{stem}.png",
        mask_dir / f"{stem}_mask.png",
        mask_dir / f"{stem}.bmp",
        mask_dir / f"{stem}.jpg",
    ]


def _find_mask(mask_dir: Path, stem: str) -> Path | None:
    for p in _mask_candidates(mask_dir, stem):
        if p.exists():
            return p
    if mask_dir.exists():
        matches = sorted(mask_dir.glob(f"{stem}*"))
        return matches[0] if matches else None
    return None


def list_mvtec_like(root: Path, category: str, split: str) -> list[ImageRecord]:
    cat_root = root / category
    records: list[ImageRecord] = []
    if split == "train":
        for p in list_images(cat_root / "train" / "good"):
            records.append(ImageRecord(p, 0, None, category, "good"))
        return records

    test_root = cat_root / "test"
    for defect_dir in sorted([p for p in test_root.iterdir() if p.is_dir()]):
        defect_type = defect_dir.name
        label = 0 if defect_type == "good" else 1
        for img in list_images(defect_dir):
            mask = None
            if label:
                mask = _find_mask(cat_root / "ground_truth" / defect_type, img.stem)
            records.append(ImageRecord(img, label, mask, category, defect_type))
    return records


def list_mvtec_ad2(root: Path, category: str, split: str) -> list[ImageRecord]:
    if category not in MVTEC_AD2_CATEGORIES:
        raise ValueError(f"Unknown MVTec AD 2 category: {category}")
    split = "test_public" if split == "test" else split
    category_root = root / category
    records: list[ImageRecord] = []
    if split in {"train", "validation"}:
        for image_path in list_images(category_root / split / "good"):
            records.append(ImageRecord(image_path, 0, None, category, "good"))
        return records
    if split == "test_public":
        for defect_type in ("good", "bad"):
            label = 0 if defect_type == "good" else 1
            for image_path in list_images(category_root / split / defect_type):
                mask_path = None
                if label:
                    candidate = (
                        category_root
                        / split
                        / "ground_truth"
                        / "bad"
                        / f"{image_path.stem}_mask.png"
                    )
                    mask_path = candidate if candidate.exists() else None
                records.append(
                    ImageRecord(
                        image_path,
                        label,
                        mask_path,
                        category,
                        defect_type,
                    )
                )
        return records
    if split in {"test_private", "test_private_mixed"}:
        for image_path in list_images(category_root / split):
            records.append(
                ImageRecord(image_path, 0, None, category, "private")
            )
        return records
    raise ValueError(f"Unsupported MVTec AD 2 split: {split}")


def list_visa(root: Path, category: str, split: str) -> list[ImageRecord]:
    if split not in {"train", "test"}:
        raise ValueError(f"Unsupported VisA split: {split}")

    official_csv = root / "split_csv" / "1cls.csv"
    if official_csv.exists():
        records: list[ImageRecord] = []
        with official_csv.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            required = {"object", "split", "label", "image", "mask"}
            missing = required - set(reader.fieldnames or [])
            if missing:
                raise ValueError(
                    f"VisA official split is missing columns {sorted(missing)}: {official_csv}"
                )
            for row in reader:
                if row["object"].strip() != category or row["split"].strip() != split:
                    continue
                label_text = row["label"].strip().lower()
                if label_text not in {"normal", "anomaly"}:
                    raise ValueError(
                        f"Unsupported VisA label '{row['label']}' in {official_csv}"
                    )
                label = 0 if label_text == "normal" else 1
                image_path = root / row["image"].strip()
                mask_rel = row["mask"].strip()
                mask_path = root / mask_rel if mask_rel else None
                if label == 0:
                    mask_path = None
                records.append(
                    ImageRecord(
                        image_path,
                        label,
                        mask_path,
                        category,
                        "good" if label == 0 else "anomaly",
                    )
                )
        return records

    raise FileNotFoundError(
        f"VisA requires the official one-class split file: {official_csv}"
    )


class IndustrialAnomalyDataset(Dataset):
    def __init__(
        self,
        records: list[ImageRecord],
        image_size: int = 256,
        max_samples: int | None = None,
        sample_seed: int | None = None,
    ) -> None:
        if max_samples is not None:
            records = _limit_records(records, max_samples, sample_seed=sample_seed)
        self.records = records
        self.image_size = image_size
        self.image_tf = transforms.Compose(
            [
                transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

    def __len__(self) -> int:
        return len(self.records)

    def _load_mask(self, rec: ImageRecord) -> torch.Tensor:
        if rec.label == 0 or rec.mask_path is None or not rec.mask_path.exists():
            return torch.zeros(1, self.image_size, self.image_size)
        mask = Image.open(rec.mask_path).convert("L")
        mask = mask.resize((self.image_size, self.image_size), Image.NEAREST)
        arr = (np.asarray(mask) > 0).astype(np.float32)
        return torch.from_numpy(arr).unsqueeze(0)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        rec = self.records[idx]
        image = self.image_tf(load_rgb(rec.image_path))
        mask = self._load_mask(rec)
        return {
            "image": image,
            "mask": mask,
            "label": torch.tensor(rec.label, dtype=torch.long),
            "path": str(rec.image_path),
            "category": rec.category,
            "defect_type": rec.defect_type,
        }


def _limit_records(
    records: list[ImageRecord],
    max_samples: int,
    sample_seed: int | None = None,
) -> list[ImageRecord]:
    limit = max(0, int(max_samples))
    if len(records) <= limit:
        return records
    labels = {int(r.label) for r in records}
    if labels != {0, 1} or limit < 2:
        if sample_seed is None:
            return records[:limit]
        rng = np.random.default_rng(int(sample_seed))
        indices = np.sort(rng.choice(len(records), size=limit, replace=False))
        return [records[int(index)] for index in indices]

    normal = [r for r in records if int(r.label) == 0]
    anomaly = [r for r in records if int(r.label) == 1]
    if sample_seed is not None:
        rng = np.random.default_rng(int(sample_seed))
        normal = [normal[int(index)] for index in rng.permutation(len(normal))]
        anomaly = [anomaly[int(index)] for index in rng.permutation(len(anomaly))]
    n_normal = min(len(normal), max(1, limit // 2))
    n_anomaly = min(len(anomaly), max(1, limit - n_normal))
    remaining = limit - n_normal - n_anomaly
    if remaining > 0:
        normal_extra = min(len(normal) - n_normal, remaining)
        n_normal += normal_extra
        remaining -= normal_extra
    if remaining > 0:
        n_anomaly += min(len(anomaly) - n_anomaly, remaining)

    selected = normal[:n_normal] + anomaly[:n_anomaly]
    return selected[:limit]


def build_dataset(
    dataset_name: str,
    root: str | Path,
    category: str,
    split: str,
    image_size: int,
    max_samples: int | None = None,
    sample_seed: int | None = None,
) -> IndustrialAnomalyDataset:
    root = Path(root)
    if dataset_name == "mvtec_ad2":
        records = list_mvtec_ad2(root, category, split)
    elif dataset_name in {"mvtec", "mpdd"}:
        records = list_mvtec_like(root, category, split)
    elif dataset_name == "visa":
        records = list_visa(root, category, split)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    if not records:
        raise FileNotFoundError(f"No records found for {dataset_name}/{category}/{split} at {root}")
    return IndustrialAnomalyDataset(
        records,
        image_size=image_size,
        max_samples=max_samples,
        sample_seed=sample_seed,
    )


class DTDTextureDataset:
    def __init__(self, root: str | Path) -> None:
        self.paths = list_images(root)

    def sample(
        self,
        size: tuple[int, int],
        rng: np.random.RandomState | None = None,
    ) -> np.ndarray:
        rng = rng or np.random
        if not self.paths:
            return rng.rand(size[1], size[0], 3).astype(np.float32)
        for _ in range(min(8, len(self.paths))):
            path = self.paths[int(rng.randint(0, len(self.paths)))]
            try:
                with Image.open(path) as img:
                    img = img.convert("RGB").resize(size, Image.BILINEAR)
                    return np.asarray(img).astype(np.float32) / 255.0
            except OSError:
                continue
        return rng.rand(size[1], size[0], 3).astype(np.float32)
