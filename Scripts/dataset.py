import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, Subset
from torchvision import transforms
from PIL import Image
from collections import Counter
import settings

#converte variações de cor pro ideal (LAB)
#conversão de RGB para LAB (L=Lightness, A=Green-Red, B=Blue-Yellow)
class ReinhardNormalizer: 
    def __init__(self):
        self.target_means = [226.60, 131.83, 121.30]
        self.target_stds = [27.74, 5.38, 8.79]
        
    def transform(self, img_pil):
        try:
            img_np = np.array(img_pil)
            
            if (img_np.shape[-1] == 4):
                img_np = img_np[:,:,:3]
                
            img_lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB).astype("float32")
            
            mu_l = np.mean(img_lab[:,:,0])
            if mu_l > 200:
                return img_pil
            
            for i in range(3):
                mu = np.mean(img_lab[:,:,i])
                sigma = np.std(img_lab[:,:,i])
                
                if (sigma == 0): sigma = 1e-5
                
                #formula de Reinhard = (Pixel - Media_Origem) * (Desvio_alvo / Desvio_Origem) + Media_alvo
                img_lab[:,:,i] = ((img_lab[:,:,i] - mu) * (self.target_stds[i] / sigma) + self.target_means[i])
            
            img_lab = np.clip(img_lab, 0, 255).astype("uint8")
            img_rgb = cv2.cvtColor(img_lab, cv2.COLOR_LAB2RGB)
            
            return Image.fromarray(img_rgb)
        
        except Exception as e:
            #retornando a orginal para não quebrar o treino
            #talvez logar o erro
            return img_pil  
        


class LeukemiaDataset(Dataset):
    def __init__(self, filepaths, labels, transform=None, use_stain_norm=False):
        """
        Args:
            filepaths: Lista de caminhos completos das imagens,
            labels: Lista de labels numéricos (0, 1, 2),
            transform: Transforms do torchvision (Resize, toTensor),
            use_stain_norm: Se for verdade, aplica Reinhard antes do transform
        """
        self.filepaths = filepaths
        self.labels = labels
        self.transform = transform
        self.use_stain_norm = use_stain_norm
        self.normalizer = ReinhardNormalizer() if use_stain_norm else None
        
    def __len__(self):
        return len(self.filepaths)
    
    def __getitem__(self, idx):
        img_path = self.filepaths[idx]
        label = self.labels[idx]
        
        try:
            image = Image.open(img_path).convert("RGB")
            
            #aplica normalização de Reinhard - se ativado no experimento
            if self.use_stain_norm and self.normalizer:
                image = self.normalizer.transform(image)
            
            #aplica resize e augmentations (x64, x128, x224, x512) 
            if self.transform:
                image = self.transform(image)
            
            return image, label
        
        except Exception as e:
            print(f"[ERRO] Erro ao ler a imagem {img_path}:{e}")
            dummy_res = 224 #tamanho seguro
            
            if self.transform:
                #tenta descobrir o tamanhp alvo pelo transform
                for t in self.transform.transforms:
                    if isinstance(t, transforms.Resize):
                        dummy_res = t.size[0]
                        break
            return torch.zeros((3, dummy_res, dummy_res)), label
        
def get_all_filepaths(pool_dir):
    """
    varre a pasta TRAIN_VAL_POOL e retorna listas de caminhos e labels
    essencial pra alimentar o k-fold
    """
    filepaths = []
    labels = []
    
    for class_name in settings.CLASSES:
        class_dir = os.path.join(pool_dir, class_name)
        class_idx = settings.CLASS_TO_IDX[class_name]
        
        if not os.path.exists(class_dir):
            print(f"[ERRO] Pasta da classe {class_name} não encontrada em {pool_dir}")
            continue
        
        files = os.listdir(class_dir)
        valid_files = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'))]            
        for f in valid_files:
            filepaths.append(os.path.join(class_dir, f))
            labels.append(class_idx)
            
    return np.array(filepaths), np.array(labels)
    
def get_transforms(resolution):
     """
     gera os pipelines de transformação para treino e validação
     recebe 'resolution' para os experimentos de 64 a 512px
     """
     train_transform = transforms.Compose([
         transforms.Resize((resolution, resolution)),
         transforms.RandomHorizontalFlip(),
         transforms.RandomVerticalFlip(),
         transforms.RandomRotation(30),
         transforms.ColorJitter(brightness=0.1, contrast=0.1),
         transforms.ToTensor(),
         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
     ])
     
     val_transform = transforms.Compose([
         transforms.Resize((resolution, resolution)),
         transforms.ToTensor(),
         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
     ])
     
     return train_transform, val_transform
    
def calculate_class_weights(labels):
    """
    calcula pesos inversos para lidar com desbalanceamento de classes
    'Wighted Cros Entropy Loss' mencionada no artigos
    tensor de pesos para passar ao loss function
    """
    
    count_dict = Counter(labels)
    total_sample = len(labels)
    num_classes = len(settings.CLASSES)
    
    weights = []
    
    for i in range(num_classes):
        count = count_dict.get(i, 0)
        if count > 0:
            w = total_sample / (num_classes * count)
        else:
            w = 1.0 #Fallback
        
        weights.append(w)
    
    return torch.tensor(weights, dtype=torch.float)