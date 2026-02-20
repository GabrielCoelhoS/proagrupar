import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
import gc
from tqdm import tqdm
import shutil 
import time

import settings
import dataset
import metrics
from cnn_mamba import HybridCNNMamba


if not os.path.exists("checkpoints"):
    os.makedirs("checkpoints")

def run_fold(train_files, train_labels, val_files, val_labels, resolution, fold_num, class_weights, exp_name):
    print(f"[INFO] Fold {fold_num} iniciando...")
    
    
    t_train, t_val = dataset.get_transforms(resolution)
    
    train_dataset = dataset.LeukemiaDataset(train_files, train_labels, transform=t_train)
    val_dataset = dataset.LeukemiaDataset(val_files, val_labels, transform=t_val)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=settings.BATCH_SIZE, 
        shuffle=True, 
        num_workers=settings.NUM_WORKERS, 
        pin_memory=settings.PIN_MEMORY
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=settings.BATCH_SIZE, 
        shuffle=False, 
        num_workers=settings.NUM_WORKERS, 
        pin_memory=settings.PIN_MEMORY
    )
    
    model = HybridCNNMamba(num_classes=settings.NUM_CLASSES).to(settings.DEVICE)
    
    criterion = nn.CrossEntropyLoss(
        weight=class_weights.to(settings.DEVICE), 
        label_smoothing=0.1
    )
    
    optimizer = optim.AdamW(model.parameters(), lr=settings.LEARNING_RATE, weight_decay=0.05)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=settings.EPOCHS)
    
    best_stats = {"acc": 0.0, "f1": 0.0}
    
    temp_model_path = f"checkpoints/{exp_name}_Fold{fold_num}_temp.pth"
    
    for epoch in range(settings.EPOCHS):
        
        model.train()
        loop_train = tqdm(
            train_loader, 
            desc=f"Epoca {epoch+1}/{settings.EPOCHS}", 
            unit="step", 
            colour="blue"
        )
        
        for imgs, labels in loop_train:
            imgs, labels = imgs.to(settings.DEVICE), labels.to(settings.DEVICE)
            
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            loop_train.set_postfix({'loss': f'{loss.item():.4f}'})
        
        model.eval()
        monitor = metrics.MetricMonitor()
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(settings.DEVICE), labels.to(settings.DEVICE)
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                monitor.update(loss.item(), outputs, labels)
        

        scheduler.step()
        
        results = monitor.get_results()
        if results["f1"] > best_stats["f1"]:
            best_stats = results
            torch.save(model.state_dict(), temp_model_path)
            
    del model, optimizer, scheduler, train_loader, val_loader
    torch.cuda.empty_cache()
    gc.collect()
    
    return best_stats

def main():
    print(f"------ INICIANDO TREINO (OTIMIZADO PARA DISCO) ------")
    print(f"Device: {settings.DEVICE}")
    print(f"Learning Rate Base: {settings.LEARNING_RATE}")
    
    timestamp = int(time.time())
    print(f"[INFO] ID da Execução (Timestamp): {timestamp}")
    
    RESOLUTIONS = [64, 128, 224]
    DATA_FRACTIONS = [0.25, 0.50, 1.0]
    
    all_files, all_labels = dataset.get_all_filepaths(settings.POOL_DIR)
    class_weight = dataset.calculate_class_weights(all_labels)
    print(f"[INFO] Pesos de Classe: {class_weight}")

    model_probe = HybridCNNMamba(num_classes=settings.NUM_CLASSES).to(settings.DEVICE)
    model_config = {
        "mamba_blocks": model_probe.mamba_depth,
        "unfrozen_backbone_tensors": model_probe.unfrozen_backbone_param_tensors,
    }
    print(
        f"[INFO] Configuração do modelo: "
        f"{model_config['mamba_blocks']} blocos Mamba | "
        f"{model_config['unfrozen_backbone_tensors']} tensores descongelados no backbone"
    )
    del model_probe
    torch.cuda.empty_cache()
    gc.collect()
    
    for res in RESOLUTIONS:
        for frac in DATA_FRACTIONS:
            exp_name = f"CNNMamba_Res{res}_Data{int(frac*100)}_{timestamp}"
            print(f"\n[INFO] >>> EXPERIMENTO: {exp_name}")
            
            best_exp_f1 = -1.0
            final_best_path = f"checkpoints/{exp_name}_BEST.pth"
            
           
            num_samples = int(len(all_files) * frac)
            np.random.seed(settings.SEED)
            indices = np.random.choice(len(all_files), num_samples, replace=False)
            exp_files, exp_labels = all_files[indices], all_labels[indices]
            
            skf = StratifiedKFold(n_splits=settings.K_FOLDS, shuffle=True, random_state=settings.SEED)
            fold_results = [] 

            for fold_i, (train_idx, val_idx) in enumerate(skf.split(exp_files, exp_labels)):

                stats = run_fold(
                    train_files=exp_files[train_idx], 
                    train_labels=exp_labels[train_idx],
                    val_files=exp_files[val_idx], 
                    val_labels=exp_labels[val_idx],
                    resolution=res, 
                    fold_num=fold_i+1, 
                    class_weights=class_weight,
                    exp_name=exp_name
                )
                
                fold_results.append(stats)
                print(f"[RESULTADO] FOLD {fold_i+1}: F1: {stats['f1']:.4f} | Acc: {stats['acc']:.4f}")
                
                temp_path = f"checkpoints/{exp_name}_Fold{fold_i+1}_temp.pth"
                
                if stats['f1'] > best_exp_f1:
                    best_exp_f1 = stats['f1']
                    print(f"   >>> Novo Recorde! ({best_exp_f1:.4f}). Promovendo a BEST model")
                    
                    if os.path.exists(temp_path):
                        shutil.move(temp_path, final_best_path)
                else:
                    print(f"   >>> Fold inferior ao recorde atual ({best_exp_f1:.4f}). Deletando temp.")
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
                
            final_stats = {}
            for metric in ["acc", "f1", "precision", "recall"]:
                values = [f[metric] for f in fold_results]
                final_stats[f"{metric}_mean"] = f"{np.mean(values):.4f}"
                final_stats[f"{metric}_std"] = f"{np.std(values):.4f}" 

            final_stats["mamba_blocks"] = model_config["mamba_blocks"]
            final_stats["unfrozen_backbone_tensors"] = model_config["unfrozen_backbone_tensors"]
                
            print(f"[FINAL] {exp_name} -> F1 Média: {final_stats['f1_mean']}")
            metrics.save_results_experiment("RESULTADOS_FINAIS.csv", exp_name, final_stats)

if __name__ == "__main__":
    main()
