"""  In this experiment we will inspect effect of augmentation given compare reports with base-model ."""

#===========import============
from pathlib import Path
import time


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

from run_experiment import run_experiment
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
import pandas as pd

import torch
import torch.nn as nn


#===========control-randomness=============
set_seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#============paths===========
csv_report_path = ROOT / "reports" / "csv" / "aug_model"
plot_report_path = ROOT / "reports" / "plots" / "aug_model"
models_path = ROOT / "models" / "aug_model"

csv_report_path.mkdir(parents=True, exist_ok=True)
plot_report_path.mkdir(parents=True, exist_ok=True)
models_path.mkdir(parents=True, exist_ok=True)

#=============Model-CNN-Base==============

class CnnAug(nn.Module):
    """CNN with augmented train_data, Same structure with base_model; class with 2 block Conv + acitvation + pooling"""
    def __init__(self,num_classes=8):
        super().__init__()

        #=====block-1=======
        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=32,
            kernel_size=3,
            bias=False,
            padding=1)

        self.norm1 = nn.BatchNorm2d(num_features=32)
        #=======block-2========
        self.conv2 = nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=3,
            padding=1,
            bias=False
        )

        self.norm2 = nn.BatchNorm2d(num_features=64)
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU()

        #========Classifier-Head=========
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(in_features=64, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=num_classes),
        )


    #==========forward==========
    def forward(self, x):
        #======layer-1=========
        out_conv1 = self.conv1(x)
        out_norm1 = self.norm1(out_conv1)
        out_relu = self.relu(out_norm1)
        out_max_pool1 = self.max_pool(out_relu)
        #=======layer-2=========
        out_conv2 = self.conv2(out_max_pool1)
        out_norm2 = self.norm2(out_conv2)
        out_relu2 = self.relu(out_norm2)
        out_max_pool2 = self.max_pool(out_relu2)
        #=======FC-layer=========
        out_fc = self.head(out_max_pool2)
        return out_fc


#=========load-data============
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


cleaner_output = data_cleaner(train_path,test_path,unclean_path)

dataset_train, dataset_train_base, dataset_test = make_clean_dataset(
    unclean_path,
    train_path,
    test_path,
    cleaner_output
)

train_loader,_, validation_loader, test_loader = make_loader(
    dataset_train,
    dataset_train_base,
    dataset_test
)


#==========train-requires-obj===========
cnn_base = CnnAug().to(DEVICE)

loss_fn = nn.CrossEntropyLoss()

optimizer = torch.optim.AdamW(cnn_base.parameters(), lr=1e-3)

classes = dataset_train.classes

start_time = time.perf_counter()
history,best_epoch,model = run_experiment(
    model=cnn_base,
    train_loader=train_loader,
    val_loader=validation_loader,
    loss_fn=loss_fn,
    optimizer=optimizer,
    device=DEVICE,
    epochs=8,
    class_names = classes
)

elapsed_time = time.perf_counter() - start_time

minutes = int(elapsed_time // 60)
seconds = elapsed_time % 60

print(f"Training completed in: {minutes}m {seconds:.2f}s ({elapsed_time:.2f}s)")
#============save-reports-->csv==============
csv_report = pd.DataFrame({
    "train_f1_score":history["train_f1"],
    "train_accuracy":history["train_accuracy"],

    "val_f1_score":history["val_f1"],
    "val_accuracy":history["val_accuracy"],

    "train_loss":history["train_loss"],
    "val_loss":history["val_loss"]
})

cf_report_val = (
    pd.DataFrame(history["val_cf_report"][best_epoch]).transpose().round(4)
)
cf_report_val.to_csv(csv_report_path/"AugModel_CF_VAL.csv",index_label="class")
cf_report_train = (
    pd.DataFrame(history["train_cf_report"][best_epoch]).transpose().round(4)
)

cf_report_train.to_csv(csv_report_path/"AugModel_CF_train.csv",index_label="class")

csv_report.to_csv(csv_report_path/"CNN-AugModel.csv",index=False)


dsp = ConfusionMatrixDisplay(
    confusion_matrix=history["val_confusion_matrix"][best_epoch],
    display_labels=dataset_train.classes)
dsp.plot(xticks_rotation=45)
dsp.ax_.set_title(f"Best Epoch: {best_epoch+1}")
plt.savefig(plot_report_path/"CNN-AugModel_val.png")

dsp = ConfusionMatrixDisplay(
    confusion_matrix=history["train_confusion_matrix"][best_epoch],
    display_labels=dataset_train.classes)
dsp.plot(xticks_rotation=45)

dsp.ax_.set_title(f"Best Epoch: {best_epoch+1}")
plt.savefig(plot_report_path/"CNN-AugModel_train.png")
plt.close()

train_images, labels = next(iter(train_loader))

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
plt.savefig(plot_report_path / "CNN-AugModel_samples.png", dpi=200)
plt.close()


torch.save(model.state_dict(),models_path/"CNN-AugModel.pth")

print("Saved Successfully")
