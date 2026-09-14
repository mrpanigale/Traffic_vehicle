"""
Preprocess images, audit data integrity, and build clean DataLoaders.
"""

# @---------------imports----------------@
import hashlib
import json
from pathlib import Path

from PIL import Image
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import ImageFolder
from sklearn.model_selection import StratifiedShuffleSplit

# @---------------Paths----------------@
ROOT = Path(__file__).resolve().parent.parent

TRAIN_PATH = ROOT / "data_set" / "train"
TEST_PATH = ROOT / "data_set" / "test"
UNCLEAN_PATH = ROOT / "data_set" / "unclean"
REPORT_DIR = ROOT / "reports" / "preprocess"


# @---------------Transforms----------------@
def get_transforms(
        image_size = (224, 224),
        augment: bool = False
):
    """
    Returns train and evaluation transforms.
    """
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]

    eval_transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
    ])

    if augment:
        train_transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
        ])
    else:
        train_transform = eval_transform

    return train_transform, eval_transform


# @---------------Auditing & Cleaning----------------@
def compute_image_hash(img_path: Path):
    """Computes hash of raw image """
    hasher = hashlib.md5()
    with open(img_path, "rb") as f:
        while chunk := f.read(8192):
            hasher.update(chunk)
    return hasher.hexdigest()


def audit_and_clean_dataset(
        train_dir: Path,
        test_dir: Path,
        unclean_dir: Path,
        save_report: bool = True
):
    """
    Audits images for corruption, size bounds, cross-set leaks, and label conflicts.
    """
    reports = {
        "train_class_counts": {},
        "min_size": None,
        "max_size": None,
        "corrupt_files": [],
        "internal_duplicates": [],
        "train_test_leaks": [],
        "train_unclean_overlap": [],
        "label_conflicts": [],
        "excluded_train_count": 0,
    }

    test_hashes = {}
    if test_dir.exists():
        for p in test_dir.rglob("*.*"):
            if p.suffix.lower() in [".jpg", ".jpeg", ".png"]:
                try:
                    with Image.open(p) as img:
                        img.load()
                    h = compute_image_hash(p)
                    test_hashes[h] = (str(p), p.parent.name)
                except Exception:
                    reports["corrupt_files"].append(str(p))

    unclean_hashes = {}
    if unclean_dir.exists():
        for p in unclean_dir.rglob("*.*"):
            if p.suffix.lower() in [".jpg", ".jpeg", ".png"]:
                try:
                    with Image.open(p) as img:
                        img.load()
                    h = compute_image_hash(p)
                    unclean_hashes[h] = (str(p), p.parent.name)
                except Exception:
                    reports["corrupt_files"].append(str(p))

    train_hashes = {}
    widths, heights = [], []
    excluded_train_files= set()

    for p in sorted(train_dir.rglob("*.*")):
        if p.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
            continue

        str_p = str(p.resolve())
        try:
            with Image.open(p) as img:
                w, h = img.size
                img.load()
            widths.append(w)
            heights.append(h)
        except Exception:
            reports["corrupt_files"].append(str_p)
            excluded_train_files.add(str_p)
            continue

        label = p.parent.name
        reports["train_class_counts"][label] = reports["train_class_counts"].get(label, 0) + 1
        img_hash = compute_image_hash(p)

        if img_hash in test_hashes:
            test_path_str, test_label = test_hashes[img_hash]
            reports["train_test_leaks"].append({
                "train_file": str_p,
                "test_file": test_path_str,
                "train_label": label,
                "test_label": test_label
            })
            if label != test_label:
                reports["label_conflicts"].append({
                    "source": "train_vs_test",
                    "train_file": str_p,
                    "other_file": test_path_str,
                    "train_label": label,
                    "other_label": test_label
                })
            excluded_train_files.add(str_p)
            continue

        if img_hash in unclean_hashes:
            unclean_path_str, unclean_label = unclean_hashes[img_hash]
            reports["train_unclean_overlap"].append({
                "train_file": str_p,
                "unclean_file": unclean_path_str
            })
            if label != unclean_label:
                reports["label_conflicts"].append({
                    "source": "train_vs_unclean",
                    "train_file": str_p,
                    "other_file": unclean_path_str,
                    "train_label": label,
                    "other_label": unclean_label
                })

        if img_hash in train_hashes:
            reports["internal_duplicates"].append({
                "file": str_p,
                "duplicate_of": train_hashes[img_hash]
            })
            excluded_train_files.add(str_p)
        else:
            train_hashes[img_hash] = str_p

    if widths and heights:
        reports["min_size"] = [min(widths), min(heights)]
        reports["max_size"] = [max(widths), max(heights)]

    reports["excluded_train_count"] = len(excluded_train_files)

    if save_report:
        REPORT_DIR.mkdir(parents=True, exist_ok=True)
        report_file = REPORT_DIR / "data_audit_report.json"
        with open(report_file, "w", encoding="utf-8") as f:
            json.dump(reports, f, indent=4)
        print(f"[Audit] Report saved -> {report_file}")

    return reports, excluded_train_files


# @---------------Data Loaders Provider----------------@
def get_dataloaders(
        train_root: Path = TRAIN_PATH,
        test_root: Path = TEST_PATH,
        excluded_files = None,
        batch_size: int = 32,
        val_size: float = 0.2,
        augment: bool = False,
        seed: int = 42
):
    """
    make dataloaders for validation and training and check assertion too.
    """
    excluded_files = excluded_files or set()
    train_tf, eval_tf = get_transforms(augment=augment)

    train_dataset_full = ImageFolder(root=train_root, transform=train_tf)
    val_dataset_full = ImageFolder(root=train_root, transform=eval_tf)

    valid_indices = [
        idx for idx, (path_str, _) in enumerate(train_dataset_full.samples)
        if str(Path(path_str).resolve()) not in excluded_files
    ]

    clean_targets = [train_dataset_full.targets[i] for i in valid_indices]

    sss = StratifiedShuffleSplit(n_splits=1, test_size=val_size, random_state=seed)
    train_sub_pos, val_sub_pos = next(sss.split(valid_indices, clean_targets))

    train_indices = [valid_indices[i] for i in train_sub_pos]
    val_indices = [valid_indices[i] for i in val_sub_pos]

    train_subset = Subset(train_dataset_full, train_indices)
    val_subset = Subset(val_dataset_full, val_indices)

    g = torch.Generator().manual_seed(seed)

    train_loader = DataLoader(
        train_subset, batch_size=batch_size, shuffle=True, generator=g, num_workers=0
    )
    val_loader = DataLoader(
        val_subset, batch_size=batch_size, shuffle=False, generator=g, num_workers=0
    )

    test_loader = None
    if test_root and test_root.exists():
        test_dataset = ImageFolder(root=test_root, transform=eval_tf)
        assert train_dataset_full.class_to_idx == test_dataset.class_to_idx, (
            f"Class mapping mismatch!\nTrain: {train_dataset_full.class_to_idx}\nTest: {test_dataset.class_to_idx}"
        )
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False, num_workers=0
        )

    return train_loader, val_loader, test_loader, train_dataset_full.classes


# @---------------Verification Block----------------@
if __name__ == "__main__":
    reports, excluded = audit_and_clean_dataset(TRAIN_PATH, TEST_PATH, UNCLEAN_PATH)
    print(f"[Audit Summary] Excluded train files: {len(excluded)}")
    print(f"[Audit Summary] Test leaks found: {len(reports['train_test_leaks'])}")
    print(f"[Audit Summary] Label conflicts: {len(reports['label_conflicts'])}")

    train_loader, val_loader, test_loader, classes = get_dataloaders(
        train_root=TRAIN_PATH,
        test_root=TEST_PATH,
        excluded_files=excluded,
        batch_size=32,
        augment=True,
        seed=42
    )

    print(f"\nClasses verified ({len(classes)}): {classes}")
    print(f"Train samples: {len(train_loader.dataset)} | Batches: {len(train_loader)}")
    print(f"Val samples: {len(val_loader.dataset)} | Batches: {len(val_loader)}")
    if test_loader:
        print(f"Test samples: {len(test_loader.dataset)} | Batches: {len(test_loader)}")

    x, y = next(iter(train_loader))
    print(f"Train batch tensor: {x.shape} | Labels: {y[:8].tolist()}")
