"""plot-few train samples from loader to see what model see"""
#=============imports=============
import torch
from data_provider import (

set_seed,
SEED,
compute_hashes,
get_balanced_sampler,
data_cleaner,
get_transform,
make_clean_dataset,
make_loader,
to_str_list,
ROOT,
unclean_path,
train_path,
test_path,
)
import matplotlib.pyplot as plt



plot_report_path = ROOT / "reports" / "plots" / "preprocess"
plot_report_path.mkdir(parents=True, exist_ok=True)
#=========load-data============

if train_path.exists():
    print("train path exists")
else:
    raise FileNotFoundError("train path does not exist")



cleaner_output = data_cleaner(train_path,test_path,unclean_path)

_, dataset_train_base, _, _ = make_clean_dataset(
    unclean_path,
    train_path,
    test_path,
    cleaner_output
)

_,train_base_loader, _, _, _ = make_loader(
    _,
    dataset_train_base,
    _,
    _
)

classes = dataset_train_base.classes
#===============plot samples=============

train_images, labels = next(iter(train_base_loader))

mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

fig, ax = plt.subplots(2, 16, figsize=(20, 3))

for image, label, axis in zip(train_images, labels, ax.flat):
    # Denormalize: X = (X_norm * std) + mean
    img_unnorm = (image * std + mean).permute(1, 2, 0).clamp(0, 1).cpu().numpy()

    axis.imshow(img_unnorm, aspect="equal")
    axis.set_title(classes[label.item()], fontsize=8)
    axis.axis("off")

plt.tight_layout()
plt.savefig(plot_report_path / "base_train_samples.png", dpi=200)
plt.close()
