"""
Preprocess images, find out duplicates and build clean DataLoaders.
"""

#@-----------------imports-----------------@
# path and torch requires
from pathlib import Path
import torch

# loaders requires -->

from torch.utils.data import DataLoader,Subset
from sklearn.model_selection import StratifiedShuffleSplit

# compute hash and open images requires -->

import hashlib
from PIL import Image
from torchvision import transforms
from torchvision.datasets import ImageFolder

#@-----------------paths-----------------@
# project-path -->
ROOT = Path(__file__).resolve().parent.parent

#sets paths -->
unclean_path = ROOT / "data_set" / "unclean"
train_path = ROOT / "data_set" / "train"
test_path = ROOT / "data_set" / "test"

#@-----------------functions-----------------@
def compute_hashes(img_path: Path):
    """compute hash code to identify images
    Return:
        hash code
    """
    # build hasher obj
    hasher = hashlib.md5()

    # open image
    with open(img_path, "rb") as img:
        # read_image gradually
        while chunk := img.read(8192):
            hasher.update(chunk)

        # hash code
        return hasher.hexdigest()





def data_cleaner(train_set_path:Path, test_set_path:Path,unclean_set_path:Path):
    """find duplicates , label conflicts, Data Leakage, Save reports """
    # check to find corrupted images
    report = {
        "corrupt":[],
        "train_hashes":[],
        "test_hashes":[],
        "unclean_hashes":[],

        "train_internal_duplicates":[],
        "test_duplicates":[],
        "unclean_duplicates":[],

        "train_internal_conflicts":[],
        "test_conflicts":[],
        "unclean_conflicts":[],
    }

    if unclean_set_path.exists():
        for path in unclean_set_path.rglob("*.*"):
            if path.suffix.lower() in [".jpg",".jpeg",".png"]:

                try:
                    with Image.open(path) as img:
                        img.load()
                    h = compute_hashes(path)

                    report["unclean_hashes"].append((h,path,path.parent.name))
                except Exception as e:
                    report["corrupt"].append(path)

    if train_set_path.exists():
        for path in train_set_path.rglob("*.*"):
            if path.suffix.lower() in [".jpg",".jpeg",".png"]:
                try:
                    with Image.open(path) as img:
                        img.load()
                    h = compute_hashes(path)
                    report["train_hashes"].append((h,path,path.parent.name))
                except Exception as e:
                    report["corrupt"].append(path)

    if test_set_path.exists():
        for path in test_set_path.rglob("*.*"):
            if path.suffix.lower() in [".jpg",".jpeg",".png"]:
                try:
                    with Image.open(path) as img:
                        img.load()
                    h = compute_hashes(path)
                    report["test_hashes"].append((h,path,path.parent.name))
                except Exception as e:
                    report["corrupt"].append(path)


    train_hash_counter = {hash_code:0 for hash_code,path,_ in report["train_hashes"]}
    test_hashes = set([hash_code for hash_code,_,_ in report["test_hashes"]])
    unclean_hashes = {hash_code:path for hash_code,path,_ in report["unclean_hashes"]}

    # key --> hash , value --> label
    test_labels = {hash_code:label for hash_code, _, label in report["test_hashes"]}
    unclean_labels = {hash_code: label for hash_code, _, label in report["unclean_hashes"]}
    train_first_seen_label = {}

    for hash_code,path,label in report["train_hashes"]:

        # train duplicates with test
        if hash_code in test_hashes:
            report["test_duplicates"].append(path)
            # append conflicts
            if label != test_labels[hash_code]:
                report["test_conflicts"].append(path)


        # internal duplicates
        if hash_code in train_hash_counter:
            #assign label to hash for first time
            if train_hash_counter[hash_code] == 0 :
                train_first_seen_label[hash_code] = label

            train_hash_counter[hash_code] += 1
            if train_hash_counter[hash_code] >= 2:
                # train path most remove from training
                report["train_internal_duplicates"].append(path)
                if label != train_first_seen_label[hash_code]:
                    report["train_internal_conflicts"].append(path)

        # unclean duplicates with train
        if hash_code in unclean_hashes:
            # unclean hash most remove from unclean --> Value == path unclean
            report["unclean_duplicates"].append(unclean_hashes[hash_code])
            if label != unclean_labels[hash_code]:
                report["unclean_conflicts"].append(unclean_hashes[hash_code])



    return report

def safe_path(train_set_path:Path, test_set_path:Path,unclean_set_path:Path,report:dict):
    """this function returns safe paths that are not duplicates and don't have conflicts"""

    pass

def get_transform():
    """build suitable transforms for train and test
    Return:
            (augmented transforms  ,  baseline transforms)
    """

    #training with augmentation transform
    aug_transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.ColorJitter(brightness=0.2,contrast=0.2),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])
    ])

    # Validation and Test transform
    base_transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])

    ])

    return aug_transform, base_transform

def make_loader():

    """make loader for train and test"""



#@-----------------Test-Block-----------------@
if __name__ == "__main__":

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

    output = data_cleaner(train_path, test_path, unclean_path)
    print(len(output["unclean_duplicates"]))
