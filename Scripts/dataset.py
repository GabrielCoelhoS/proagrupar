import os
import numpy as np
import torch
from torch.utils.data import Dataset, Subset
from torchvision import transforms
from PIL import Image
from collections import Counter
import settings

# --- CLASSE REINHARD REMOVIDA ---
# Decisão de Design: Devido à alta heterogeneidade do dataset UNION (18 fontes),
# a normalização estática estava introduzindo artefatos (cores neon/fundo verde).
# Optamos por deixar a CNN aprender a invariância de cor via Data Augmentation.

class LeukemiaDataset(Dataset):
    def __init__(self, filepaths, labels, transform=None, use_stain_norm=False):
        """
        Args:
            filepaths: Lista de caminhos
            labels: Lista de labels
            transform: Transforms do torchvision
            use_stain_norm: MANTIDO APENAS POR COMPATIBILIDADE, MAS NÃO FAZ NADA AGORA.
        """
        self.filepaths = filepaths
        self.labels = labels
        self.transform = transform
        # Ignoramos a flag use_stain_norm propositalmente
        
    def __len__(self):
        return len(self.filepaths)
    
    def __getitem__(self, idx):
        img_path = self.filepaths[idx]
        label = self.labels[idx]
        
        try:
            # 1. Carrega a imagem original (RAW)
            image = Image.open(img_path).convert("RGB")
            
            # 2. Aplica apenas Transforms (Resize + Augmentations padrão)
            if self.transform:
                image = self.transform(image)
            
            return image, label
        
        except Exception as e:
            print(f"[ERRO] Erro ao ler a imagem {img_path}: {e}")
            dummy_res = 224
            if self.transform:
                for t in self.transform.transforms:
                    if isinstance(t, transforms.Resize):
                        dummy_res = t.size[0]
                        break
            return torch.zeros((3, dummy_res, dummy_res)), label

# --- Funções Auxiliares ---

def get_all_filepaths(pool_dir):
    filepaths = []
    labels = []
    
    for class_name in settings.CLASSES:
        class_dir = os.path.join(pool_dir, class_name)
        class_idx = settings.CLASS_TO_IDX[class_name]
        
        if not os.path.exists(class_dir):
            print(f"[ERRO] Pasta {class_name} não encontrada.")
            continue
        
        files = os.listdir(class_dir)
        valid_files = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'))]            
        for f in valid_files:
            filepaths.append(os.path.join(class_dir, f))
            labels.append(class_idx)
            
    return np.array(filepaths), np.array(labels)
    
def get_transforms(resolution):
     """
     Gera os pipelines.
     AQUI ESTÁ O SEGREDO: Usamos ColorJitter para ensinar a rede a lidar com cores diferentes,
     em vez de tentar consertar as cores antes.
     """
     train_transform = transforms.Compose([
         transforms.Resize((resolution, resolution)),
         transforms.RandomHorizontalFlip(),
         transforms.RandomVerticalFlip(),
         transforms.RandomRotation(30),
         
         # DATA AUGMENTATION DE COR (Substitui o Reinhard)
         # Ensinamos a rede: "Não ligue se a imagem for um pouco mais clara, escura ou saturada"
         transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02),
         
         transforms.ToTensor(),
         # Normalização padrão do ImageNet (Média/Std matemáticos dos pixels)
         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
     ])
     
     val_transform = transforms.Compose([
         transforms.Resize((resolution, resolution)),
         transforms.ToTensor(),
         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
     ])
     
     return train_transform, val_transform
    
def calculate_class_weights(labels):
    count_dict = Counter(labels)
    total_sample = len(labels)
    num_classes = len(settings.CLASSES)
    weights = []
    
    for i in range(num_classes):
        count = count_dict.get(i, 0)
        if count > 0:
            w = total_sample / (num_classes * count)
        else:
            w = 1.0
        weights.append(w)
    
    return torch.tensor(weights, dtype=torch.float)
