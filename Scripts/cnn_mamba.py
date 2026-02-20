import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import math


class VSSBlock(nn.Module):
    def __init__(self, d_model, d_state=16, expand=2, dt_rank="auto"):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.expand = expand
        self.d_inner = int(expand * d_model)
        self.dt_rank = math.ceil(d_model / 16) if dt_rank == "auto" else dt_rank
        
        #projeções de entrada
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        
        #convolução local - simula visal local antes do SSM
        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            kernel_size=3,
            padding=1,
            bias=True
        )
        self.act = nn.SiLU()
        
        #Projeções dos parametros SSM (A, B, C, Delta) 
        
        # x_proj: Delta (dt_rank) + B (d_state) + C (d_state)
        self.x_proj = nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False)
        
        #dt_proj: delta para dimensão interna
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)
        
        #inicialização do parametro A - logaritmico para estabilidade
        # A_log - "fator esquecimento" de memoria
        A = torch.arange(1, self.d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        
        #projeção de saida
        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        
    def selective_scan_pure_pytorch(self, u, delta, A, B, C, D):
        """
        implementação pura em pytorch do scan seletivo (sem CUDA kernels)
        
        u: input [Batch, Lenght, Dim]
        delta: passo de tempo [Batch, lenght, Dim]
        A: matriz de estado [Dim, State]
        B: entrada de controle [Batch, Lenght, State]
        C: saida de controle [Batch, Lenght, State]
        """
        batch, lenght, dim = u.shape
        d_state = A.shape[1]
        
        #discretização (A continuo -> A discreto)
        #deltaA = exp(delta * A)
        deltaA = torch.exp(torch.einsum('bld, dn->bldn', delta, A))
        
        #discretização de B (deltaB = delta * B)
        deltaB_u = torch.einsum('bld,bln,bld->bldn', delta, B, u)
        
        #scan - loop de recorencia 
        # h_t = A_bar * h_{t-1} + B_bar * x_t 
        h = torch.zeros(batch, dim, d_state, device=u.device)
        ys = []
        
        for t in range(lenght):
            #atualiza o estado oculto de h
            h = deltaA[:, t] * h + deltaB_u[:, t]
            #calcula saida y = C * h
            y = torch.einsum('bdn,bn->bd', h, C[:, t])
            ys.append(y)
            
        y = torch.stack(ys, dim=1) #[Batch, Length, Dim]
        
        #adiçao da conexão residual D * u
        y = y + u * D 
        return y
    
    def forward(self, x):
        #x input: [Batch, Lenght, Dim] - achatado pela MobileNet
        #x entra como sequencia
        
        #projeção  linear + chunk - divide em x e z
        #z é usado pra o "Gating" final - multiplicação
        u_z = self.in_proj(x)
        u, z = u_z.chunk(2, dim=-1) #u e z: [B, L, D_inner]
        
        #convulução 2d - transforma a sequencia de volta pra imagem 2D
        #224px -> 7x7 (L=49), 512px -> 16x16 (L=256) 
        B, L, D = u.shape
        H = W = int(math.sqrt(L))
        
        u_img = u.transpose(1, 2).view(B, D, H, W)
        u_img = self.conv2d(u_img)
        u_img = self.act(u_img)
        u = u_img.view(B, D, L).transpose(1, 2) #Volta pra [B, L, D]
        
        # calculo dos parametros seletivos - Delta, B, C
        x_dbl = self.x_proj(u) #projeção para obter parametros
        delta, B_ssm, C_ssm = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        
        #projeção de delta
        delta = F.softplus(self.dt_proj(delta))
        
        #Recupera A negativo
        A = -torch.exp(self.A_log)
        
        #execuçao do Scan Seletivo
        y = self.selective_scan_pure_pytorch(u, delta, A, B_ssm, C_ssm, self.D)
        
        #Gatting e saida
        y = y * F.silu(z)
        y = self.out_norm(y)
        out = self.out_proj(y)
        
        return out + x #Skip connection global
    
    
class HybridCNNMamba(nn.Module):
    def __init__(self, num_classes, depth=4):
        super().__init__()
        backbone = models.mobilenet_v2(weights='IMAGENET1K_V1')
        self.features = backbone.features
        self.mamba_depth = depth
        
        for i, param in enumerate(self.features.parameters()):
            if (i < 100):
                param.requires_grad = False
            else:
                param.requires_grad = True

        self.total_backbone_param_tensors = sum(1 for _ in self.features.parameters())
        self.frozen_backbone_param_tensors = sum(1 for p in self.features.parameters() if not p.requires_grad)
        self.unfrozen_backbone_param_tensors = sum(1 for p in self.features.parameters() if p.requires_grad)
                
        self.cnn_channels = 1280
        self.mamba_dim = 192
        
        self.adapter = nn.Linear(self.cnn_channels, self.mamba_dim)
        
        #pilha de blocos mamba - 2 Blocos para dar profundidade
        self.layers = nn.ModuleList([
            VSSBlock(d_model=self.mamba_dim, d_state=16)
            for _ in range(depth)
        ]) 
        
        
        self.norm_f = nn.LayerNorm(self.mamba_dim)
         
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(self.mamba_dim, num_classes)
        )
        
    def forward(self, x):
        #CNN Feature Extration
        x = self.features(x) #[B, 1280, H, W]
        
        #Flatten para sequencia
        b, c, h, w = x.shape
        x = x.view(b, c, h * w).permute(0, 2, 1) #[B, L, 1280]
         
        x = self.adapter(x)
        
        #Mamba core
        for layer in self.layers:
            x = layer(x)
        
        #normalização final
        x = self.norm_f(x)
            
        #pooling e classificação
        x = x.mean(dim=1)
        return self.classifier(x)
