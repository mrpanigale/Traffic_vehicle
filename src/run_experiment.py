"""
This file runs various experiments
and returns the loss and accuracy for training and validation and Confusion matrix  in CSV format,
the trained model, and so on.
"""

#=========imports
from pathlib import Path
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

def run_epoch(model,loss_fn,loader,device,optimizer=None):
    """This function runs one epoch of training and validation."""
    is_training = optimizer is not None
    model.to(device)
    model.train(is_training)
    total_loss = 0.0
    total_correct = 0
    total_examples = 0
    all_logits = []
    all_labels = []
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

    return {
        "loss":total_loss/total_examples,
        "accuracy":total_correct/total_examples,
        "cm":confusion_matrix(
            y_true=all_labels,
            y_pred = all_logits,
            labels=list(range(10))
)
    }

def run_experiment(model,loss_fn,train_loader,val_loader,device,optimizer=None,epoch = 8):
    """This function runs experiments and return the reports"""
    history= {
        "train_loss":[],
        "train_accuracy":[],
        "train_confusion_matrix":[],
        "val_loss":[],
        "val_accuracy":[],
        "val_confusion_matrix":[]
    }

    for epoch in range(epoch):
        # for train
        train_metrics = run_epoch(
            model,
            loss_fn,
            train_loader,
            device,
            optimizer=optimizer
        )
        #for validation
        val_metrics = run_epoch(
            model,
            loss_fn,
            val_loader,
            device
        )

        history["train_accuracy"].append(train_metrics["accuracy"])
        history["train_loss"].append(train_metrics["loss"])
        history["train_confusion_matrix"].append(train_metrics["cm"])

        history["val_accuracy"].append(val_metrics["accuracy"])
        history["val_loss"].append(val_metrics["loss"])
        history["val_confusion_matrix"].append(val_metrics["cm"])

    return history