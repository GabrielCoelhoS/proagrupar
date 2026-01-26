import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os
import glob
import torchvision.transforms as transforms

import settings
import dataset
from cnn_mamba import HybridCNNMamba

def normalize_for_plot(img_tensor):
    #desfaz a normalização do pytorch pra exibir a imagem corretamente
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = img_tensor.cpu().numpy().transpose(1, 2, 0)
    img = std * img + mean
    return img

def main():
    print("----- GERANDO DIAGRAMA DE FLUXO DA ARQUITETURA -----")
    search_path = os.path.join(settings.POOL_DIR, "HBS", "*")
    files = glob.glob(search_path)
    if not files:
        search_path = os.path.join(settings.POOL_DIR, "*", "*")
        files = glob.glob(search_path)
        
    img_path = files[0]
    print(f"Imagem selecionada: {img_path}")
    
    print("Aplicando Normalização de Reinhard...")
    original_pil = Image.open(img_path).convert("RGB")
    
    reinhard = dataset.ReinhardNormalizer()
    
    normalized_pil = reinhard.transform(original_pil) 
    resolution = 224
    t_viz = transforms.Compose([
        transforms.Resize((resolution, resolution)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    input_tensor = t_viz(normalized_pil).unsqueeze(0).to(settings.DEVICE) #[1, 3, 224, 224]
    
    #Inicializa o Modelo
    model = HybridCNNMamba(num_classes=3).to(settings.DEVICE)
    model.eval()
    
    with torch.no_grad():
        print("Extraindo Features da CNN...")
        features_cnn = model.features(input_tensor) #features_cnn shape:[1, 1280, 7, 7]
        
        #"mapa de calor" medio da cnn 
        heatmap_cnn = torch.mean(features_cnn, dim=1).squeeze().cpu().numpy()
        
        #flatting - "estacamento"
        print("Transformando em Sequencia(Mamba input)...")
        b, c, h, w = features_cnn.shape
        # .view() - "transforma quadrado em linha"
        sequence_input = features_cnn.view(b, c, h * w).permute(0, 2, 1) #sequence_input shape: [1, 49, 1280]
        
        #processamento Mamba
        print("Passando pelos blocos mamba...")
        x= model.adapter(sequence_input)
        x_mamba = model.mamba1(x)
        x_mamba = model.mamba2(x_mamba) 
        #x_mamba shape: [1, 49, 192]
        
        #classificação 
        x_pool = x_mamba.mean(dim=1)
        logits = model.classifier(x_pool)
        probs = torch.nn.functional.softmax(logits, dim=1).cpu().numpy()[0]
        
    #plotagem visual
    print("Gerando gráfico...")
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle(f'Fluxo da arquitetura Hibrida CNN-Mamba\n {os.path.basename(img_path)}', fontsize=16)
    
    #original vs reinhard
    ax1 = fig.add_subplot(2, 4, 1)
    ax1.imshow(original_pil)
    ax1.set_title("1. Original (Raw)")
    ax1.axis('off')
    
    ax2 = fig.add_subplot(2, 4, 2)
    ax2.imshow(normalized_pil)
    ax2.set_title("2. Reinhard (Normalizada)") 
    ax2.axis('off')
    
    #visao da cnn
    ax3 = fig.add_subplot(2, 4, 3)
    img_bg = normalize_for_plot(input_tensor.squeeze())
    ax3.imshow(img_bg, alpha=0.6)
    ax3.imshow(heatmap_cnn, cmap='jet', alpha=0.5)
    ax3.set_title(f'3. CNN features ({h}x{w} patches)')
    ax3.axis('off')
    
    #visualização da sequencia - mamba
    ax4 = fig.add_subplot(2, 4, 4)
    mamba_vis = x_mamba.squeeze().cpu().numpy().T #Transposição para caber melhor
    ax4.imshow(mamba_vis, aspect='auto', cmap='viridis')
    ax4.set_title('4. Sequencia Mamba (49 patches)')
    ax4.set_xlabel('Posição na Sequencia (Espaço)')
    ax4.set_ylabel('Embeddings (Features)')
    
    #classificação Final 
    ax5 = fig.add_subplot(2, 1, 2) #ocupa a parte de baixo toda
    classes = settings.CLASSES
    colors = ['green' if i == np.argmax(probs) else 'gray' for i in range(3)]
    bars = ax5.bar(classes, probs * 100, color=colors)
    ax5.set_ylim(0, 100)
    ax5.set_ylabel('Probabilidade (%)')
    ax5.set_title('5. Classificação Final')  
        
    #adiciona a % nas barras
    for bar in bars:
        height = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width()/2., height, f'{height:.1f}%', ha='center', va='bottom')
        
    plt.tight_layout()
    plt.savefig("arquitetura_visual7.png")
    print("Salvo como 'arquitetura_visual7.png' ")
    
if __name__ == "__main__":
    main() 