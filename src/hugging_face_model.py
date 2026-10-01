"""We will load , fine tune and save a model from hugging face to add to predict.py"""

# =========import============
import time
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
from data_provider import (
    set_seed,
    SEED,
    data_cleaner,
    make_clean_dataset,
    make_loader,
    ROOT,
    unclean_path,
    train_path,
    test_path,
)

import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
import pandas as pd

import torch
from transformers import (
    AutoImageProcessor,
    AutoModelForImageClassification,
)

classes = [
    "ambulance",
    "autobus",
    "kamyun",
    "kamyunet",
    "minibus",
    "savari",
    "taxi",
    "vanet",
]

# ===========control-randomness=============
set_seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ============paths===========
csv_report_path = ROOT / "reports" / "csv" / "hugging_face_model"
plot_report_path = ROOT / "reports" / "plots" / "hugging_face_model"
models_path = ROOT / "models" / "hugging_face_model"

csv_report_path.mkdir(parents=True, exist_ok=True)
plot_report_path.mkdir(parents=True, exist_ok=True)
models_path.mkdir(parents=True, exist_ok=True)


# ========class==========
class HFImageDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, processor):
        self.dataset = dataset
        self.processor = processor
        self.samples = dataset.samples
        self.targets = dataset.targets

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        image, label = self.dataset[index]

        inputs = self.processor(image, return_tensors="pt")

        pixel_values = inputs["pixel_values"].squeeze(0)

        return pixel_values, label


# =============load-model=============
MODEL_ID = "apple/mobilevit-small"
NUM_CLASSES = len(classes)
# ==============labels=============
id2label = {idx: class_name for idx, class_name in enumerate(classes)}

label2id = {class_name: idx for idx, class_name in enumerate(classes)}

# =============preprocessor=========
preprocessor = AutoImageProcessor.from_pretrained(MODEL_ID)

# =============model=========
model = AutoModelForImageClassification.from_pretrained(
    MODEL_ID,
    num_labels=NUM_CLASSES,
    id2label=id2label,
    label2id=label2id,
    ignore_mismatched_sizes=True,
)

model.to(DEVICE)

print("<Hugging Face model loaded>")
print(f"<device: {DEVICE}>")
print(f"<classes: {classes}>")


# =========freeze-backbone=========
for param in model.mobilevit.parameters():
    param.requires_grad = False

# آخرین encoder layer
for param in model.mobilevit.encoder.layer[4].parameters():
    param.requires_grad = True

# classifier
for param in model.classifier.parameters():
    param.requires_grad = True
# =========load-data=========
report = data_cleaner(train_path, test_path, unclean_path)


train_dataset, train_base_dataset, test_dataset = make_clean_dataset(
    unclean_path,
    train_path,
    test_path,
    report,
    hf=True,
)


train_dataset = HFImageDataset(train_dataset, preprocessor)
train_base_dataset = HFImageDataset(train_base_dataset, preprocessor)
test_dataset = HFImageDataset(test_dataset, preprocessor)


train_loader, train_base_loader, validation_loader, test_loader = make_loader(
    train_dataset,
    train_base_dataset,
    test_dataset,
)


# =========training=========
EPOCHS = 10
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4

optimizer = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
)

history = {
    "train_f1": [],
    "train_accuracy": [],
    "val_f1": [],
    "val_accuracy": [],
    "train_loss": [],
    "val_loss": [],
    "train_cf_report": [],
    "val_cf_report": [],
    "train_confusion_matrix": [],
    "val_confusion_matrix": [],
}

best_epoch = 0
best_val_loss = float("inf")
best_model_state = None

start_time = time.perf_counter()

for epoch in range(EPOCHS):

    # ================== train ==================
    model.train()

    train_loss = 0.0
    train_predictions = []
    train_labels = []

    for images, labels in train_loader:

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(
            pixel_values=images,
            labels=labels,
        )

        loss = outputs.loss
        logits = outputs.logits

        loss.backward()
        optimizer.step()

        train_loss += loss.item()

        predictions = logits.argmax(dim=1)

        train_predictions.extend(predictions.cpu().tolist())
        train_labels.extend(labels.cpu().tolist())

    train_loss /= len(train_loader)

    train_accuracy = accuracy_score(
        train_labels,
        train_predictions,
    )

    train_f1 = f1_score(
        train_labels,
        train_predictions,
        average="weighted",
        zero_division=0,
    )

    train_cf_report = classification_report(
        train_labels,
        train_predictions,
        target_names=classes,
        output_dict=True,
        zero_division=0,
    )

    train_confusion_matrix = confusion_matrix(
        train_labels,
        train_predictions,
    )

    # ================== validation ==================
    model.eval()

    val_loss = 0.0
    val_predictions = []
    val_labels = []

    with torch.no_grad():

        for images, labels in validation_loader:

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(
                pixel_values=images,
                labels=labels,
            )

            loss = outputs.loss
            logits = outputs.logits

            val_loss += loss.item()

            predictions = logits.argmax(dim=1)

            val_predictions.extend(predictions.cpu().tolist())
            val_labels.extend(labels.cpu().tolist())

    val_loss /= len(validation_loader)

    val_accuracy = accuracy_score(
        val_labels,
        val_predictions,
    )

    val_f1 = f1_score(
        val_labels,
        val_predictions,
        average="weighted",
        zero_division=0,
    )

    val_cf_report = classification_report(
        val_labels,
        val_predictions,
        target_names=classes,
        output_dict=True,
        zero_division=0,
    )

    val_confusion_matrix = confusion_matrix(
        val_labels,
        val_predictions,
    )

    # ================== history ==================
    history["train_loss"].append(train_loss)
    history["train_accuracy"].append(train_accuracy)
    history["train_f1"].append(train_f1)

    history["val_loss"].append(val_loss)
    history["val_accuracy"].append(val_accuracy)
    history["val_f1"].append(val_f1)

    history["train_cf_report"].append(train_cf_report)
    history["val_cf_report"].append(val_cf_report)

    history["train_confusion_matrix"].append(train_confusion_matrix)
    history["val_confusion_matrix"].append(val_confusion_matrix)

    # ================== best model ==================
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_epoch = epoch
        best_model_state = {
            key: value.detach().cpu().clone()
            for key, value in model.state_dict().items()
        }

    print(
        f"Epoch [{epoch + 1}/{EPOCHS}] | "
        f"Train Loss: {train_loss:.4f} | "
        f"Train Acc: {train_accuracy:.4f} | "
        f"Train F1: {train_f1:.4f} | "
        f"Val Loss: {val_loss:.4f} | "
        f"Val Acc: {val_accuracy:.4f} | "
        f"Val F1: {val_f1:.4f}"
    )

elapsed_time = time.perf_counter() - start_time

minutes = int(elapsed_time // 60)
seconds = elapsed_time % 60

print(f"Training completed in: " f"{minutes}m {seconds:.2f}s ({elapsed_time:.2f}s)")

print(f"Best epoch: {best_epoch + 1}")
print(f"Best validation loss: {best_val_loss:.4f}")


# =========save-reports=========
csv_report = pd.DataFrame(
    {
        "train_f1_score": history["train_f1"],
        "train_accuracy": history["train_accuracy"],
        "val_f1_score": history["val_f1"],
        "val_accuracy": history["val_accuracy"],
        "train_loss": history["train_loss"],
        "val_loss": history["val_loss"],
    }
)

csv_report.to_csv(
    csv_report_path / "MobileViT_training_history.csv",
    index=False,
)

cf_report_val = pd.DataFrame(history["val_cf_report"][best_epoch]).transpose().round(4)

cf_report_val.to_csv(
    csv_report_path / "MobileViT_best_val_classification_report.csv",
    index_label="class",
)

cf_report_train = (
    pd.DataFrame(history["train_cf_report"][best_epoch]).transpose().round(4)
)

cf_report_train.to_csv(
    csv_report_path / "MobileViT_best_train_classification_report.csv",
    index_label="class",
)


# =========save-confusion-matrices=========

dsp = ConfusionMatrixDisplay(
    confusion_matrix=history["val_confusion_matrix"][best_epoch],
    display_labels=classes,
)

dsp.plot(xticks_rotation=45)
dsp.ax_.set_title(f"MobileViT - Best Epoch: {best_epoch + 1}")

plt.tight_layout()
plt.savefig(plot_report_path / "MobileViT_best_val_confusion_matrix.png")
plt.close()


dsp = ConfusionMatrixDisplay(
    confusion_matrix=history["train_confusion_matrix"][best_epoch],
    display_labels=classes,
)

dsp.plot(xticks_rotation=45)
dsp.ax_.set_title(f"MobileViT - Best Epoch: {best_epoch + 1}")

plt.tight_layout()
plt.savefig(plot_report_path / "MobileViT_best_train_confusion_matrix.png")
plt.close()


# =========save-best-model=========
best_model_path = models_path / "best_model"
best_model_path.mkdir(parents=True, exist_ok=True)

model.load_state_dict(best_model_state)

model.save_pretrained(best_model_path)
preprocessor.save_pretrained(best_model_path)

print(f"Best model saved to: {best_model_path}")
