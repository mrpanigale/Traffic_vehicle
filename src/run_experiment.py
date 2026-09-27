"""
This file runs various experiments
and returns the loss and accuracy for training and validation and Confusion matrix  in CSV format,
the trained model, and so on.
"""

#=========imports==========
from pathlib import Path
import copy
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from data_provider import set_seed,SEED
import seaborn as sns
from sklearn.metrics import confusion_matrix,f1_score,classification_report

set_seed(SEED)
#==============functions===============
def run_epoch(model,loss_fn,loader,device,optimizer=None,class_names=None):
    """This function runs one epoch of training and validation."""
    is_training = optimizer is not None
    model.to(device)

    model.eval()

    if is_training:
        for module in getattr(model, "trainable_modules", [model]):
            module.train()


    total_loss = 0.0
    total_correct = 0
    total_examples = 0
    all_logits = []
    all_labels = []
    all_preds = []
    with torch.set_grad_enabled(is_training):
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            loss = loss_fn(logits, labels)

            if is_training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_examples += labels.size(0)
            total_loss += loss.item() * labels.size(0)
            # Sum True = Total correct
            total_correct += (logits.argmax(dim=1) == labels).sum().item()
            all_logits.extend(logits.argmax(dim=1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)

    return {
        "loss":total_loss/total_examples,
        "accuracy":total_correct/total_examples,
        "f1":f1_score(
            all_labels,
            all_preds,
            average="macro",
            zero_division=0
        ),
        "cm":confusion_matrix(
            y_true=all_labels,
            y_pred = all_logits,
            labels=list(range(8))
        ),
        "classification_report": classification_report(
            y_true=all_labels,
            y_pred=all_preds,
            target_names=class_names,
            zero_division=0,
            output_dict=True)

    }

def run_experiment(
        model,
        loss_fn,
        train_loader,
        val_loader,
        device,
        optimizer=None,
        epochs = 8,
        class_names =None,
        scheduler=None
):

    """This function runs experiments and return the reports"""
    history= {
        "train_loss":[],
        "train_accuracy":[],
        "train_f1":[],
        "train_confusion_matrix":[],
        "train_cf_report":[],

        "val_loss":[],
        "val_accuracy":[],
        "val_f1":[],
        "val_confusion_matrix":[],
        "val_cf_report":[]
    }

    #=====early-stopping======
    best_val_loss = float("inf")
    best_epoch = -1
    best_model_weight = None


    for epoch in range(epochs):
        # for train
        train_metrics = run_epoch(
            model,
            loss_fn,
            train_loader,
            device,
            optimizer=optimizer,
            class_names = class_names
        )
        #for validation
        val_metrics = run_epoch(
            model,
            loss_fn,
            val_loader,
            device,
            class_names=class_names
        )
        if scheduler:
            scheduler.step()
        history["train_f1"].append(train_metrics["f1"])
        history["train_accuracy"].append(train_metrics["accuracy"])
        history["train_loss"].append(train_metrics["loss"])
        history["train_confusion_matrix"].append(train_metrics["cm"])
        history["train_cf_report"].append(train_metrics["classification_report"])

        history["val_f1"].append(val_metrics["f1"])
        history["val_accuracy"].append(val_metrics["accuracy"])
        history["val_loss"].append(val_metrics["loss"])
        history["val_confusion_matrix"].append(val_metrics["cm"])
        history["val_cf_report"].append(val_metrics["classification_report"])

        val_loss = val_metrics["loss"]

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_model_weight = copy.deepcopy(model.state_dict())

    if best_model_weight is not None:
        model.load_state_dict(best_model_weight)
    return history,best_epoch,model