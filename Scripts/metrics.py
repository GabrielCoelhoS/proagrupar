import torch
import numpy as np
import csv
import os
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, f1_score, precision_score, recall_score

class MetricMonitor:
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.val_loss = 0
        self.val_acc = 0
        self.count = 0
        self.all_preds = []
        self.all_targets = []
        
    def update(self, loss, outputs, targets):
        #salva as predições e alvos para calcular metricas globais no final
        _, preds = torch.max(outputs, 1)
        self.all_preds.extend(preds.cpu().numpy())
        self.all_targets.extend(targets.cpu().numpy())
        
        #loss e acc istantaneos
        batch_size = outputs.size(0)
        self.val_loss += loss * batch_size
        correct = (preds == targets).sum().item()
        self.val_acc += correct
        self.count += batch_size
        
    def get_results(self):
        avg_loss = self.val_loss / self.count
        avg_acc = self.val_acc / self.count
        
        f1 = f1_score(self.all_targets, self.all_preds, average='macro', zero_division=0)
        prec = precision_score(self.all_targets, self.all_preds, average='macro', zero_division=0)
        rec = recall_score(self.all_targets, self.all_preds, average='macro', zero_division=0)
        
        return {
            "loss": avg_loss,
            "acc": avg_acc,
            "f1": f1,
            "precision": prec,
            "recall": rec
        }
        
    
def save_results_experiment(filename, experiment_name, results_dict):
    """
    salva uma linha no CSV com todas as metricas
    results_dict: conter as medias e desvios padrão (acc_mean, acc_std)
    """

    file_exists = os.path.isfile(filename)
    columns = ['Experimento'] + list(results_dict.keys())
    
    with open(filename, mode='a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        if not file_exists:
            writer.writeheader()
            
        row = {'Experimento': experiment_name}
        row.update(results_dict)
        writer.writerow(row)

def save_per_class(filename, targets_test, preds_test, class_names):
    report = classification_report(
    targets_test, preds_test, target_names=class_names, output_dict=True, zero_division=0
    )

    # Apenas linhas das classes
    rows = []
    for cls in class_names:
        row = report[cls]
        rows.append({
            "class": cls,
            "precision": row["precision"],
            "recall": row["recall"],
            "f1": row["f1-score"],
            "support": row["support"],
        })



    file_exists = os.path.isfile(filename)
    
    with open(filename, mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=["class", "precision", "recall", "f1", "support"])
        if not file_exists:
            writer.writeheader()

        writer.writerows(rows)


def save_overall_metrics(filename, targets_test, preds_test):
    report = classification_report(
        targets_test, preds_test, output_dict=True, zero_division=0
    )

    row = {
        "accuracy": report["accuracy"],
        "macro_precision": report["macro avg"]["precision"],
        "macro_recall": report["macro avg"]["recall"],
        "macro_f1": report["macro avg"]["f1-score"],
        "weighted_precision": report["weighted avg"]["precision"],
        "weighted_recall": report["weighted avg"]["recall"],
        "weighted_f1": report["weighted avg"]["f1-score"],
    }

    file_exists = os.path.isfile(filename)
    with open(filename, mode='a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

    return row


def save_confusion_matrix_csv(filename, targets_test, preds_test, class_names):
    cm = confusion_matrix(targets_test, preds_test, labels=list(range(len(class_names))))

    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["true\\pred"] + class_names)
        for i, class_name in enumerate(class_names):
            writer.writerow([class_name] + cm[i].tolist())

    return cm


def save_confusion_matrix_plot(filename, targets_test, preds_test, class_names):
    cm = confusion_matrix(targets_test, preds_test, labels=list(range(len(class_names))))

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    fig.colorbar(im, ax=ax)

    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predito")
    ax.set_ylabel("Verdadeiro")
    ax.set_title("Matriz de Confusao")

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center", color="black")

    fig.tight_layout()
    fig.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return cm
