"""
Preprocess images and data-providing for other unit
"""

#@---------------imports----------------@
import hashlib
import json
from collections import Counter
from pathlib import Path
from PIL import Image

import torch
from torch.utils.data import DataLoader, Subset
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from sklearn.model_selection import StratifiedShuffleSplit


#@---------------Paths----------------@
root = Path(__file__).resolve().parent.parent

train_path =  root / "data_set" / "train"
test_path = root / "data_set" / "test"
unclean_path = root / "data_set" / "unclean"
#@---------------functions----------------@
def get_tranforms(image_size=(224, 224)):
    """get transforms return suitable transforms for train with augmentation and validation(baseline)"""
    # @---------------transforms----------------@
    train_transform = transforms.Compose(
            [
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
            ]
        )

    base_transforms = transform=transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])
    ])

    return train_transform, base_transforms


def compute_image_hash(img_path: Path):
    """This function compute and return image hash to check and find duplicate images"""

    with open(img_path,"rb") as f:
        return hashlib.md5(f.read()).hexdigest()

def audit_clean_dataset(train_dir:Path,test_dir:Path):
    """This file will check duplicates and ..."""


    valid_train_files = []
    test_hashes = {}
    train_hashes = {}
    cross_duplicates = []
    internal_duplicates = []

    for file_path in test_dir.rglob("*.*"):
        if file_path.suffix.lower() in [".jpg", ".jpeg", ".png"]:
            try:
                with Image.open(file_path) as img:
                    img.verify()
                test_hashes[compute_image_hash(file_path)] = file_path
            except Exception as e:
                print(f"Test image corrupt: {file_path.name} | Error: {e}")

    for file_path in train_dir.rglob("*.*"):
        if file_path.suffix.lower() in [".jpg", ".jpeg", ".png"]:
            try:
                with Image.open(file_path) as img:
                    img.verify()
            except Exception as e:
                print(f"Train image corrupt: {file_path.name} | Error: {e}")
                continue

            h = compute_image_hash(file_path)
            label = file_path.parent.name

            if h in test_hashes:
                cross_duplicates.append((file_path, test_hashes[h]))
            elif h in train_hashes:
                internal_duplicates.append((file_path, train_hashes[h]))
            else:
                train_hashes[h] = (file_path, label)
                valid_train_files.append(file_path)

    print(f"Verified & Unique Train: {len(valid_train_files)}")
    print(f"Cross Duplicates (Train/Test Leak): {len(cross_duplicates)}")
    print(f"Internal Duplicates in Train: {len(internal_duplicates)}")

    return valid_train_files, {"cross": cross_duplicates, "internal": internal_duplicates}


def get_loader(dataset_root:Path,batch_size:int=32,seed=int(42)):
    ...
#@---------------test-block----------------@
if __name__=="__main__":
    audit_clean_dataset(train_path,test_path)

