""" Base-Line CNN Model + average pooling effect. """

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
csv_report_path = ROOT / "reports" / "csv" / "avgpool_model"
plot_report_path = ROOT / "reports" / "plots" / "avgpool_model"
models_path = ROOT / "models" / "avgpool_model"

csv_report_path.mkdir(parents=True, exist_ok=True)
plot_report_path.mkdir(parents=True, exist_ok=True)
models_path.mkdir(parents=True, exist_ok=True)

#=============Model-CNN-Average-Pool==============
class CnnAvgPool(nn.Module):
    """CNN with 4 Conv blocks and AvgPool2d (Ablation study on Pooling method)."""

    def __init__(self, num_classes=8):
        super().__init__()

        # ===== Block 1 =====
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False)
        self.norm1 = nn.BatchNorm2d(32)

        # ===== Block 2 =====
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False)
        self.norm2 = nn.BatchNorm2d(64)

        # ===== Block 3 =====
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False)
        self.norm3 = nn.BatchNorm2d(128)

        # ===== Block 4 =====
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False)
        self.norm4 = nn.BatchNorm2d(256)

        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU()

        # ===== Classifier Head =====
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(in_features=256, out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=num_classes),
        )

    def forward(self, x):
        # Layer 1
        x = self.pool(self.relu(self.norm1(self.conv1(x))))
        # Layer 2
        x = self.pool(self.relu(self.norm2(self.conv2(x))))
        # Layer 3
        x = self.pool(self.relu(self.norm3(self.conv3(x))))
        # Layer 4
        x = self.pool(self.relu(self.norm4(self.conv4(x))))

        # FC Head
        return self.head(x)

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

_,train_base_loader, validation_loader, test_loader = make_loader(
    dataset_train,
    dataset_train_base,

    dataset_test
)


#==========train-requires-obj===========
cnn_base = CnnAvgPool().to(DEVICE)

loss_fn = nn.CrossEntropyLoss()

optimizer = torch.optim.AdamW(cnn_base.parameters(), lr=1e-3)

classes = dataset_train.classes

start_time = time.perf_counter()
history,best_epoch,model = run_experiment(
    model=cnn_base,
    train_loader=train_base_loader,
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
cf_report_val.to_csv(csv_report_path/"avgPoolModel_CF_VAL.csv",index_label="class")
cf_report_train = (
    pd.DataFrame(history["train_cf_report"][best_epoch]).transpose().round(4)
)

cf_report_train.to_csv(csv_report_path/"avgPoolModel_CF_train.csv",index_label="class")

csv_report.to_csv(csv_report_path/"CNN-avgPoolModel.csv",index=False)


dsp = ConfusionMatrixDisplay(
    confusion_matrix=history["val_confusion_matrix"][best_epoch],
    display_labels=dataset_train.classes)
dsp.plot(xticks_rotation=45)
dsp.ax_.set_title(f"Best Epoch: {best_epoch+1}")
plt.savefig(plot_report_path/"CNN-avgPoolModel_val.png")

dsp = ConfusionMatrixDisplay(
    confusion_matrix=history["train_confusion_matrix"][best_epoch],
    display_labels=dataset_train.classes)
dsp.plot(xticks_rotation=45)

dsp.ax_.set_title(f"Best Epoch: {best_epoch+1}")
plt.savefig(plot_report_path/"CNN-avgPoolModel_train.png")
plt.close()


torch.save(model.state_dict(),models_path/"CNN-avgPoolModel.pth")
print("Saved Successfully")
