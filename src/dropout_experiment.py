""" Test dropout with 2 different value 0.3 and 0.5  on base model ."""

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
csv_report_path = ROOT / "reports" / "csv" / "dropout_model"
plot_report_path = ROOT / "reports" / "plots" / "dropout_model"
models_path = ROOT / "models" / "dropout_model"

csv_report_path.mkdir(parents=True, exist_ok=True)
plot_report_path.mkdir(parents=True, exist_ok=True)
models_path.mkdir(parents=True, exist_ok=True)

#=============Model-CNN-Base==============
import torch.nn as nn


class CnnDropout(nn.Module):
    """CNN with 4 Conv blocks and Dropout in Classifier Head (Ablation study on Regularization)."""

    def __init__(self, num_classes=8, dropout_rate=0.3):
        super().__init__()

        # ===== Block 1 =====
        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=32,
            kernel_size=3,
            bias=False,
            padding=1,
        )
        self.norm1 = nn.BatchNorm2d(num_features=32)

        # ===== Block 2 =====
        self.conv2 = nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=3,
            bias=False,
            padding=1,
        )
        self.norm2 = nn.BatchNorm2d(num_features=64)

        # ===== Block 3 =====
        self.conv3 = nn.Conv2d(
            in_channels=64,
            out_channels=128,
            kernel_size=3,
            bias=False,
            padding=1,
        )
        self.norm3 = nn.BatchNorm2d(num_features=128)

        # ===== Block 4 =====
        self.conv4 = nn.Conv2d(
            in_channels=128,
            out_channels=256,
            kernel_size=3,
            bias=False,
            padding=1,
        )
        self.norm4 = nn.BatchNorm2d(num_features=256)

        # ===== Utils =====
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU()

        # ===== Classifier Head =====
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(in_features=256, out_features=256),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features=256, out_features=num_classes),
        )

    def forward(self, x):
        # Layer 1
        out_conv1 = self.conv1(x)
        out_norm1 = self.norm1(out_conv1)
        out_relu1 = self.relu(out_norm1)
        out_max_pool1 = self.max_pool(out_relu1)

        # Layer 2
        out_conv2 = self.conv2(out_max_pool1)
        out_norm2 = self.norm2(out_conv2)
        out_relu2 = self.relu(out_norm2)
        out_max_pool2 = self.max_pool(out_relu2)

        # Layer 3
        out_conv3 = self.conv3(out_max_pool2)
        out_norm3 = self.norm3(out_conv3)
        out_relu3 = self.relu(out_norm3)
        out_max_pool3 = self.max_pool(out_relu3)

        # Layer 4
        out_conv4 = self.conv4(out_max_pool3)
        out_norm4 = self.norm4(out_conv4)
        out_relu4 = self.relu(out_norm4)
        out_max_pool4 = self.max_pool(out_relu4)

        # Classifier Head
        out_fc = self.head(out_max_pool4)
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

_,train_base_loader, validation_loader, test_loader = make_loader(
    dataset_train,
    dataset_train_base,
    dataset_test
)


#==========train-requires-obj===========



classes = dataset_train.classes
dropouts=[0.3,0.5]
for rate in dropouts:
    run_name = f"dropout_{int(rate*10)}"
    cnn_dropout = CnnDropout(num_classes=len(classes),dropout_rate=rate).to(DEVICE)
    loss_fn = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(cnn_dropout.parameters(), lr=1e-3)
    start_time = time.perf_counter()
    history,best_epoch,model = run_experiment(
        model=cnn_dropout,
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
    cf_report_val.to_csv(csv_report_path/f"DropoutModel_CF_VAL{run_name}.csv",index_label="class")
    cf_report_train = (
        pd.DataFrame(history["train_cf_report"][best_epoch]).transpose().round(4)
    )

    cf_report_train.to_csv(csv_report_path/f"DropoutModel_CF_train{run_name}.csv",index_label="class")

    csv_report.to_csv(csv_report_path/f"CNN-DropoutModel{run_name}.csv",index=False)


    dsp = ConfusionMatrixDisplay(
        confusion_matrix=history["val_confusion_matrix"][best_epoch],
        display_labels=dataset_train.classes)
    dsp.plot(xticks_rotation=45)
    dsp.ax_.set_title(f"Best Epoch: {best_epoch+1}")
    plt.savefig(plot_report_path/f"CNN-DropoutModel_val{run_name}.png")

    dsp = ConfusionMatrixDisplay(
        confusion_matrix=history["train_confusion_matrix"][best_epoch],
        display_labels=dataset_train.classes)
    dsp.plot(xticks_rotation=45)

    dsp.ax_.set_title(f"Best Epoch: {best_epoch+1}")
    plt.savefig(plot_report_path/f"CNN-DropoutModel_train{run_name}.png")
    plt.close()


    torch.save(model.state_dict(),models_path/f"CNN-DropoutModel{run_name}.pth")
print("Saved Successfully")
