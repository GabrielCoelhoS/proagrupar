import torch
import numpy as np
import csv
import os
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
            
        row = {'Experimento': {experiment_name}}
        row.update(results_dict)
        writer.writerow(row)