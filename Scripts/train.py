import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedKFold
import gc
from tqdm import tqdm

import settings
import dataset
import metrics
from cnn_mamba import HybridCNNMamba

def run_fold(train_idx, val_idx, full_dataset, resolution, fold_num, class_weights):
    print(f"[INFO] Fold {fold_num} iniciando treino")
    
    train_subset = Subset(full_dataset, train_idx)
    val_subset = Subset(full_dataset, val_idx)
    
    train_loader = DataLoader(train_subset, batch_size=settings.BATCH_SIZE, shuffle=True, num_workers=settings.NUM_WORKERS, pin_memory=settings.PIN_MEMORY)
    val_loader = DataLoader(val_subset, batch_size=settings.BATCH_SIZE, shuffle=False, num_workers=settings.NUM_WORKERS, pin_memory=settings.PIN_MEMORY)
    
    model = HybridCNNMamba(num_classes=settings.NUM_CLASSES).to(settings.DEVICE)
    
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(settings.DEVICE))
    optimizer = optim.AdamW(model.parameters(), lr=settings.LEARNING_RATE, weight_decay=0.05)
    
    best_stats = {"acc": 0.0, "f1": 0.0}
    
    for epoch in range(settings.EPOCHS):
        # print(f"[INFO] Epoch {epoch+1}/{settings.EPOCHS}...", end='\r')
        #treino 
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
        
        #validação    
        model.eval()
        monitor = metrics.MetricMonitor()
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(settings.DEVICE), labels.to(settings.DEVICE)
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                monitor.update(loss.item(), outputs, labels)
                
        results = monitor.get_results()
        if(results["f1"] > best_stats["f1"]):
            best_stats = results
    
    #faxina na memoria
    del model, optimizer, train_loader, val_loader
    torch.cuda.empty_cache()
    gc.collect()
    
    return best_stats

#loop principal de experimentos 
def main():
    print(f"------ INICIANDO TREINO ------")
    print(f"Device: {settings.DEVICE}")
    
    RESOLUTIONS = [64, 128, 224, 512]
    DATA_FRACTIONS = [0.25, 0.50, 1.0]
    
    all_files, all_labels = dataset.get_all_filepaths(settings.POOL_DIR)
    class_weight = dataset.calculate_class_weights(all_labels)
    print(f"[INFO] Pessos calculados para loss:{class_weight}")
    
    for res in RESOLUTIONS:
        for frac in DATA_FRACTIONS:
            exp_name = f"CNNMamba_Res{res}_Data{int(frac*100)}"
            print(f"[INFO] EXPERIMENTO: {exp_name}")
            
            num_samples = int(len(all_files) * frac)
            np.random.seed(settings.SEED)
            indices = np.random.choice(len(all_files), num_samples, replace=False)
            exp_files, exp_labels = all_files[indices], all_labels[indices]
            
            #Preparação do Dataset - transform muda de acordo com a resolução
            t_train, t_val = dataset.get_transforms(res)
            full_dataset = dataset.LeukemiaDataset(exp_files, exp_labels, transform=t_train, use_stain_norm=True)
            
            #inicialização do k-fold  
            skf = StratifiedKFold(n_splits=settings.K_FOLDS, shuffle=True, random_state=settings.SEED)
            fold_results = [] 

            for fold_i, (train_idx, val_idx) in enumerate(skf.split(exp_files, exp_labels)):
                stats = run_fold(train_idx, val_idx, full_dataset, res, fold_i+1, class_weight)
                fold_results.append(stats)
                print(f"[EXPERIMENTO] FOLD {fold_i+1} -> F1: {stats['f1']:.4f} | Acc: {stats['acc']:.4f}")
                
            final_stats = {}
            for metric in ["acc", "f1", "precision", "recall"]:
                values = [f[metric] for f in fold_results]
                final_stats[f"{metric}_mean"] = f"{np.mean(values):.4f}"
                final_stats[f"{metric}_std"] = f"{np.std(values):.4f}" 
                
            print(f"[FINAL] >>> FINAL {exp_name}: F1 Média = {final_stats['f1_mean']}")
            metrics.save_results_experiment("RESULTADOS_FINAIS.csv", exp_name, final_stats)
            
if __name__ == "__main__":
    main()