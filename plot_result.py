import psutil
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    precision_recall_curve,
    auc
)
import time
import matplotlib.pyplot as plt
import seaborn as sns


def evaluate(y_true, y_pred, labels=None, measure_time=False, measure_memory=False):
    if labels is None:
        labels = list(set(y_true))  # Dynamically handle labels if not provided

    mapping = {label: idx for idx, label in enumerate(labels)}

    def map_func(x):
        return mapping.get(x, -1)  # Map to -1 if not found, should not occur with correct data

    y_true_mapped = np.vectorize(map_func)(y_true)
    y_pred_mapped = np.vectorize(map_func)(y_pred)

    # Measure inference time if required
    start_time = time.time()
    if measure_time:
        for _ in range(100):  # Simulate repeated inference for more reliable timing
            _ = np.vectorize(map_func)(y_pred)
    inference_time = time.time() - start_time if measure_time else None

    # Measure memory usage if required
    memory_usage = None
    if measure_memory:
        process = psutil.Process()
        memory_usage = process.memory_info().rss / (1024 ** 2)  # Convert bytes to MB

    # Accuracy
    accuracy = accuracy_score(y_true=y_true_mapped, y_pred=y_pred_mapped)

    # Accuracy per label
    unique_labels = set(y_true_mapped)
    label_accuracies = {}
    for label in unique_labels:
        label_indices = [i for i in range(len(y_true_mapped)) if y_true_mapped[i] == label]
        label_y_true = [y_true_mapped[i] for i in label_indices]
        label_y_pred = [y_pred_mapped[i] for i in label_indices]
        label_accuracy = accuracy_score(label_y_true, label_y_pred)
        label_accuracies[labels[label]] = label_accuracy

    # Classification report
    class_report = classification_report(
        y_true=y_true_mapped,
        y_pred=y_pred_mapped,
        target_names=labels,
        labels=list(range(len(labels))),
        output_dict=True  # Return as dictionary for better parsing
    )

    # Confusion matrix
    conf_matrix = confusion_matrix(
        y_true=y_true_mapped,
        y_pred=y_pred_mapped,
        labels=list(range(len(labels)))
    )

    # ROC-AUC and PR-AUC (for binary classification)
    if len(unique_labels) == 2:
        auc_roc = roc_auc_score(y_true_mapped, y_pred_mapped)
        precision, recall, _ = precision_recall_curve(y_true_mapped, y_pred_mapped)
        auc_pr = auc(recall, precision)
    else:
        auc_roc, auc_pr = None, None  # Not applicable for multi-class

    # Create a dictionary of metrics
    metrics = {
        "overall_accuracy": accuracy,
        "label_accuracies": label_accuracies,
        "classification_report": class_report,
        "confusion_matrix": conf_matrix,
        "roc_auc": auc_roc,
        "pr_auc": auc_pr,
        "inference_time": inference_time,
        "memory_usage": memory_usage
    }

    return metrics
