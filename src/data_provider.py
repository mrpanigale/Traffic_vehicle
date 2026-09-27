"""
Preprocess images, find out duplicates and build clean DataLoaders.
"""

# @-----------------imports-----------------@
# path and torch requires
from pathlib import Path
import json
import torch
import numpy as np
import random

# loaders requires -->
from torch.utils.data import WeightedRandomSampler
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split

# compute hash and open images requires -->
from collections import defaultdict
import imagehash
from PIL import Image
from torchvision import transforms
from torchvision.datasets import ImageFolder

# @-----------------paths-----------------@
# project-path -->
ROOT = Path(__file__).resolve().parent.parent

# sets paths -->
unclean_path = ROOT / "data_set" / "unclean"
train_path = ROOT / "data_set" / "train"
test_path = ROOT / "data_set" / "test"
report_path = ROOT / "reports" / "preprocess"
SEED = 42


# @-----------------classes-----------------@
class RGBImageFolder(ImageFolder):
    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path).convert("RGB")
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, target


# @-----------------functions-----------------@
def set_seed(seed=SEED):
    """control random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_hashes(img_path: Path):
    """compute perceptual hash to identify visually similar images"""
    with Image.open(img_path).convert("L") as img:
        hash_obj = imagehash.phash(img)
        return str(hash_obj)

    # open image
    with open(img_path, "rb") as img:
        # read_image gradually
        while chunk := img.read(8192):
            hasher.update(chunk)

        # hash code
        return hasher.hexdigest()


def get_balanced_sampler(dataset):
    if isinstance(dataset, Subset):
        all_targets = np.array(dataset.dataset.targets)
        targets = torch.tensor(all_targets[dataset.indices])
    else:
        targets = torch.tensor(dataset.targets)

    class_counts = torch.bincount(targets)
    class_weights = 1.0 / class_counts.float()

    sample_weights = [class_weights[t] for t in targets]

    sampler = WeightedRandomSampler(
        weights=sample_weights, num_samples=len(sample_weights), replacement=True
    )
    return sampler


def data_cleaner(train_set_path: Path, test_set_path: Path, unclean_set_path: Path):
    """find duplicates , label conflicts, Data Leakage, Save reports"""
    # check to find corrupted images
    report = {
        "corrupt": [],
        "train_hashes": [],
        "test_hashes": [],
        "unclean_hashes": [],
        "train_internal_duplicates": [],
        "test_duplicates": [],
        "unclean_duplicates": [],
        "train_internal_conflicts": [],
        "test_conflicts": [],
        "unclean_conflicts": [],
    }

    if unclean_set_path.exists():
        for path in unclean_set_path.rglob("*.*"):
            if path.suffix.lower() in [".jpg", ".jpeg", ".png"]:

                try:
                    with Image.open(path) as img:
                        img.load()
                    h = compute_hashes(path)

                    report["unclean_hashes"].append((h, path, path.parent.name))
                except Exception as e:
                    report["corrupt"].append(path)

    if train_set_path.exists():
        for path in train_set_path.rglob("*.*"):
            if path.suffix.lower() in [".jpg", ".jpeg", ".png"]:
                try:
                    with Image.open(path) as img:
                        img.load()
                    h = compute_hashes(path)
                    report["train_hashes"].append((h, path, path.parent.name))
                except Exception as e:
                    report["corrupt"].append(path)

    if test_set_path.exists():
        for path in test_set_path.rglob("*.*"):
            if path.suffix.lower() in [".jpg", ".jpeg", ".png"]:
                try:
                    with Image.open(path) as img:
                        img.load()
                    h = compute_hashes(path)
                    report["test_hashes"].append((h, path, path.parent.name))
                except Exception as e:
                    report["corrupt"].append(path)

    # each hash code has counter= 0
    train_hash_counter = {hash_code: 0 for hash_code, path, _ in report["train_hashes"]}
    # test hashe codes to check duplicates between train and test
    test_hashes = set([hash_code for hash_code, _, _ in report["test_hashes"]])

    # path and label for each hash code for unclean set
    unclean_hashes = defaultdict(list)
    for hash_code, path, label in report["unclean_hashes"]:
        unclean_hashes[hash_code].append((path, label))

    # unclean_hashes = {hash_code:path for hash_code,path,_ in report["unclean_hashes"]}

    # key --> hash , value --> label
    test_labels = {hash_code: label for hash_code, _, label in report["test_hashes"]}

    # unclean_labels = {hash_code: label for hash_code, _, label in report["unclean_hashes"]}

    # will be used for internal duplicates
    train_first_seen_label = {}

    # run a for loop in train_hashes , labels, paths
    for hash_code, path, label in report["train_hashes"]:

        # train duplicates with test
        if hash_code in test_hashes:
            report["test_duplicates"].append(path)
            # append conflicts
            if label != test_labels[hash_code]:
                report["test_conflicts"].append(path)

        # internal duplicates
        if hash_code in train_hash_counter:
            # assign label to hash for first time
            if train_hash_counter[hash_code] == 0:
                train_first_seen_label[hash_code] = (path, label)

            train_hash_counter[hash_code] += 1
            if train_hash_counter[hash_code] >= 2:
                # train path most remove from training
                report["train_internal_duplicates"].append(path)
                first_path, first_label = train_first_seen_label[hash_code]

                if label != first_label:
                    report["train_internal_conflicts"].append(path)
                    report["train_internal_conflicts"].append(first_path)

        # unclean duplicates with train
        if hash_code in unclean_hashes:
            for unclean_hash, unclean_label in unclean_hashes[hash_code]:
                report["unclean_duplicates"].append(unclean_hash)
                if label != unclean_label:
                    report["unclean_conflicts"].append(unclean_hash)  # اینجا تصحیح شد

    return report


def get_transform():
    """build suitable transforms for train and test
    Return:
            (augmented transforms  ,  baseline transforms)
    """

    # training with augmentation transform
    aug_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # Validation and Test transform
    base_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    return aug_transform, base_transform


def make_clean_dataset(
    unclean_set_path: Path, train_set_path: Path, test_set_path: Path, report: dict
):
    bad_paths = set(
        report["corrupt"]
        + report["test_duplicates"]
        + report["unclean_duplicates"]
        + report["train_internal_duplicates"]
        + report["test_conflicts"]
        + report["unclean_conflicts"]
        + report["train_internal_conflicts"]
    )

    # check paths
    bad_paths = {p.resolve() for p in bad_paths}

    # check paths function
    def is_safe(file_path: str):
        return Path(file_path).resolve() not in bad_paths

    aug_transform, base_transform = get_transform()

    train = RGBImageFolder(
        root=train_set_path, transform=aug_transform, is_valid_file=is_safe
    )

    unclean = RGBImageFolder(
        root=unclean_set_path, transform=aug_transform, is_valid_file=is_safe
    )

    train_base = RGBImageFolder(
        root=train_set_path, transform=base_transform, is_valid_file=is_safe
    )

    unclean_base = RGBImageFolder(
        root=unclean_set_path, transform=base_transform, is_valid_file=is_safe
    )

    assert train.class_to_idx == unclean.class_to_idx, "Class mapping mismatch!"

    train.samples.extend(unclean.samples)
    train.targets.extend(unclean.targets)

    train_base.samples.extend(unclean_base.samples)
    train_base.targets.extend(unclean_base.targets)

    test = RGBImageFolder(
        root=test_set_path, transform=base_transform, is_valid_file=is_safe
    )

    return train, train_base, test


def make_loader(train_set, train_base_set, test_set, balanced=True):
    """make loader for train and test"""

    targets = [label for _, label in train_set.samples]

    train_idx, val_idx = train_test_split(
        np.arange(len(targets)),
        test_size=0.20,
        shuffle=True,
        stratify=targets,
        random_state=42,
    )

    train_subset = Subset(train_set, train_idx)
    val_subset = Subset(train_base_set, val_idx)
    if balanced:
        sampler = get_balanced_sampler(train_subset)
    train_base_subset = Subset(train_base_set, train_idx)
    if balanced:
        train_base_loader = DataLoader(
            train_base_subset,
            sampler=sampler,
            batch_size=32,
            shuffle=False,
            num_workers=0,
        )
        train_loader = DataLoader(
            train_subset, sampler=sampler, batch_size=32, shuffle=False, num_workers=0
        )
    else:
        train_base_loader = DataLoader(
            train_base_subset, batch_size=32, shuffle=True, num_workers=0
        )
        train_loader = DataLoader(
            train_subset, batch_size=32, shuffle=True, num_workers=0
        )
    validation_loader = DataLoader(
        val_subset, batch_size=64, shuffle=False, num_workers=0
    )

    test_loader = DataLoader(test_set, batch_size=64, shuffle=False, num_workers=0)

    return train_loader, train_base_loader, validation_loader, test_loader


def to_str_list(xs):
    return [str(p) for p in xs]


# @-----------------Test-Block-----------------@
if __name__ == "__main__":
    set_seed(SEED)
    # check if paths exist?
    if unclean_path.exists():
        print("unclean path exists")
    else:
        raise FileNotFoundError("unclean path does not exist")

    if train_path.exists():
        print("train path exists")
    else:
        raise FileNotFoundError("train path does not exist")

    if test_path.exists():
        print("test path exists")
    else:
        raise FileNotFoundError("test path does not exist")

    # out put data cleaner
    cleaner_output = data_cleaner(train_path, test_path, unclean_path)
    # output make clean dataset
    dataset_train, dataset_train_base, dataset_test = make_clean_dataset(
        unclean_path, train_path, test_path, cleaner_output
    )
    # output make loader
    train_loader, train_base_loader, validation_loader, test_loader = make_loader(
        dataset_train, dataset_train_base, dataset_test
    )

    train_image, train_label = next(iter(train_loader))
    train_b_image, train_b_label = next(iter(train_base_loader))

    validation_image, validation_label = next(iter(validation_loader))
    test_image, test_label = next(iter(test_loader))

    print("train_image", train_image.shape)
    print("train_label", train_label.shape)

    print("train_base_image", train_b_image.shape)
    print("train_base_label", train_b_label.shape)
    print("validation_image", validation_image.shape)
    print("validation_label", validation_label.shape)
    print("test_image", test_image.shape)
    print("test_label", test_label.shape)

    # save report
    report_json = {
        "corrupt": to_str_list(cleaner_output["corrupt"]),
        "train_internal_duplicates": to_str_list(
            cleaner_output["train_internal_duplicates"]
        ),
        "unclean_duplicates": to_str_list(cleaner_output["unclean_duplicates"]),
        "test_duplicates": to_str_list(cleaner_output["test_duplicates"]),
        "train_internal_conflicts": to_str_list(
            cleaner_output["train_internal_conflicts"]
        ),
        "unclean_conflicts": to_str_list(cleaner_output["unclean_conflicts"]),
        "test_conflicts": to_str_list(cleaner_output["test_conflicts"]),
    }

    report_path = report_path / "data_audit_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report_json, f, indent=4)
    print(f"[Audit] Report saved -> {report_json}")
