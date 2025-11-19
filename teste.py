import os
import pandas as pd
import numpy as np
import tensorflow as tf
import time
from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint
from tensorflow.keras.applications import MobileNetV2, VGG19, VGG16
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess
from tensorflow.keras.applications.vgg19 import preprocess_input as vgg19_preprocess
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg16_preprocess
from tensorflow.keras import models, layers
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import StratifiedKFold
import sys
import gc

# --- CONFIGURAÇÕES PARA MÁXIMA VELOCIDADE ---
BATCH_SIZE = 32  # Tente 64. Se der erro de memória, diminua para 32.
EPOCHS = 20      
LEARNING_RATE = 0.0002
DROPOUT_RATE = 0.2  # <--- ADICIONADO (Estava faltando)
K_FOLDS = 5

# Caminhos (formato Linux para seu WSL)
TEST_DIR = "/mnt/c/Users/algue/Documents/Pro Agrupar/UNION FOLDS/TEST_SET"
POOL_DIR = "/mnt/c/Users/algue/Documents/Pro Agrupar/UNION FOLDS/TRAIN_VAL_POOL"

CLASSES = ["HBS", "ALL", "AML"]
TARGET_SIZE = (224, 224)

# --- 1. SELEÇÃO DO MODELO ---
list_CNNs = ['MobileNetV2', 'VGG19', 'VGG16']
i = 1
for CNN in list_CNNs:
    print(f"Digite {i} para usar {CNN}")
    i += 1
number_CNN = int(input("Escolha o modelo (1, 2, ou 3): "))
name_model = "" 

if number_CNN == 1:
    selected_preprocess = mobilenet_preprocess
    name_model = "MobileNetV2"
elif number_CNN == 2:
    selected_preprocess = vgg19_preprocess 
    name_model = "VGG19"
elif number_CNN == 3:
    selected_preprocess = vgg16_preprocess 
    name_model = "VGG16"
else:
    raise ValueError(f"Modelo desconhecido: {number_CNN}")

print(f"Modelo selecionado: {name_model}")


# --- 2. CAMADA DE AUMENTO DE DADOS (Roda na GPU!) ---
data_augmentation_layers = tf.keras.Sequential([
    layers.RandomRotation(0.2),      
    layers.RandomZoom(0.2),          
    layers.RandomFlip("horizontal"), 
    layers.RandomFlip("vertical"),   
    layers.RandomContrast(0.2),      
])


# --- 3. FUNÇÕES DO PIPELINE TF.DATA ---

def load_image(filepath, label):
    """Carrega, decodifica e redimensiona a imagem."""
    image = tf.io.read_file(filepath)
    image = tf.io.decode_image(image, channels=3, expand_animations=False)
    image = tf.image.resize(image, TARGET_SIZE)
    
    # Aplica a normalização específica do modelo
    image = tf.cast(image, tf.float32)
    image = selected_preprocess(image)
    
    return image, label

def prepare_dataset(filepaths, labels, is_training=True):
    """Cria um pipeline de dados otimizado (SEM CACHE DE RAM)."""
    # 1. Cria o dataset a partir das listas de arquivos
    ds = tf.data.Dataset.from_tensor_slices((filepaths, labels))
    
    # 2. Carrega as imagens em paralelo
    ds = ds.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
    
    # --- REMOVIDO: ds = ds.cache() ---
    # Isso consumia toda a RAM e causava o erro "Killed".
    # Removendo, o script lerá do disco, que é mais seguro.
    
    if is_training:
        # 4. Embaralha (Buffer reduzido para economizar RAM)
        ds = ds.shuffle(buffer_size=500) 
        # 5. Cria os lotes (batches)
        ds = ds.batch(BATCH_SIZE)
        # 6. Aplica aumento de dados (GPU)
        ds = ds.map(lambda x, y: (data_augmentation_layers(x, training=True), y), 
                    num_parallel_calls=tf.data.AUTOTUNE)
    else:
        # Validação/Teste
        ds = ds.batch(BATCH_SIZE)
    
    # 7. Prefetch (Mantemos isso pois ajuda na velocidade sem gastar muita RAM)
    ds = ds.prefetch(buffer_size=tf.data.AUTOTUNE)
    
    return ds


# --- 4. CARREGAMENTO E PREPARAÇÃO DOS DADOS ---
print(f"Carregando lista de arquivos de : {POOL_DIR}")
all_filepaths = []
all_labels = []
class_to_int = {class_name: i for i, class_name in enumerate(CLASSES)}

for class_name in CLASSES:
    class_dir = os.path.join(POOL_DIR, class_name)
    try:
        filenames = os.listdir(class_dir)
        for f in filenames:
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif')):
                all_filepaths.append(os.path.join(class_dir, f))
                all_labels.append(class_to_int[class_name])
    except FileNotFoundError:
            print(f"[AVISO] Pasta de classe não encontrada: {class_dir}")

if not all_filepaths:
    print(f"[ERRO FATAL] Nenhum arquivo encontrado.")
    sys.exit()

x_pool = np.array(all_filepaths)
y_pool = np.array(all_labels)

print(f"Total de {len(x_pool)} imagens carregadas.")

# Cria diretórios se não existirem
if not os.path.exists('models'): os.makedirs('models')
if not os.path.exists('logs'): os.makedirs('logs')


# --- 5. FUNÇÃO DE CRIAÇÃO DO MODELO ---
def create_model_from_selection(selection):
    if selection == 1: 
        base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(TARGET_SIZE[0], TARGET_SIZE[1], 3))
    elif selection == 2: 
        base_model = VGG19(weights='imagenet', include_top=False, input_shape=(TARGET_SIZE[0], TARGET_SIZE[1], 3))
    elif selection == 3:
        base_model = VGG16(weights='imagenet', include_top=False, input_shape=(TARGET_SIZE[0], TARGET_SIZE[1], 3))
    
    base_model.trainable = False
    
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(DROPOUT_RATE), 
        layers.Dense(len(CLASSES), activation='softmax')
    ])
    
    opt = Adam(learning_rate=LEARNING_RATE)
    
    model.compile(
        optimizer=opt, 
        loss='sparse_categorical_crossentropy', 
        metrics=['accuracy']
    )
    return model


# --- 6. TREINAMENTO K-FOLD (OTIMIZADO) ---
print(f"\nIniciando K-Fold (K={K_FOLDS}) com tf.data Pipeline...")

skf = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=42)
fold_scores = []
best_fold_model_paths = []
fold_number = 1

for train_indices, val_indices in skf.split(x_pool, y_pool):
    print("\n" + "="*30)
    print(f"INICIANDO FOLD {fold_number}/{K_FOLDS}")
    print("=" * 30)

    x_train, y_train = x_pool[train_indices], y_pool[train_indices]
    x_val, y_val = x_pool[val_indices], y_pool[val_indices]

    # --- CRIA OS DATASETS OTIMIZADOS ---
    train_ds = prepare_dataset(x_train, y_train, is_training=True)
    val_ds = prepare_dataset(x_val, y_val, is_training=False)

    # Cria modelo
    model = create_model_from_selection(number_CNN)

    timestamp = int(time.time())
    caminho_log = f"logs/treino_{name_model}_fold{fold_number}_{timestamp}.csv"
    caminho_modelo = f"models/BEST_{name_model}_fold{fold_number}.keras"
    
    callbacks = [
        CSVLogger(caminho_log),
        ModelCheckpoint(caminho_modelo, monitor='val_accuracy', save_best_only=True, mode='max', verbose=1)
    ]

    print(f"Treinando (Cache ativado após 1ª época)...")
    history = model.fit(
        train_ds,
        epochs=EPOCHS,
        validation_data=val_ds,
        verbose=1,
        callbacks=callbacks
    )
    
    # Pega o melhor resultado
    best_acc = np.max(history.history['val_accuracy'])
    print(f"Melhor Acurácia Fold {fold_number}: {best_acc * 100:.2f}%")
    fold_scores.append(best_acc)
    best_fold_model_paths.append(caminho_modelo)
    
    fold_number += 1
    tf.keras.backend.clear_session() # Limpa o grafo do TensorFlow
    del model                        # Deleta a variável do Python
    gc.collect()

# --- 7. RESUMO ---
print("\n" + "=" * 30)
print(f"K-FOLD CONCLUÍDO")
if fold_scores:
    print(f"Média: {np.mean(fold_scores)*100:.2f}% (+/- {np.std(fold_scores)*100:.2f}%)")
    best_kfold_score = np.max(fold_scores)
else:
    best_kfold_score = 0


# --- 8. PROVA FINAL (OTIMIZADA) ---
print("\n" + "=" * 40)
print(f"--- PROVA FINAL ---")
print("=" * 40)

# Prepara dataset final (todo o POOL)
final_train_ds = prepare_dataset(x_pool, y_pool, is_training=True)

# Prepara dataset de Teste
test_filepaths = []
test_labels = []
for class_name in CLASSES:
    c_dir = os.path.join(TEST_DIR, class_name)
    try:
        for f in os.listdir(c_dir):
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif')):
                test_filepaths.append(os.path.join(c_dir, f))
                test_labels.append(class_to_int[class_name])
    except: pass

if not test_filepaths:
    print("[ERRO] Não encontrou imagens de teste.")
else:
    test_ds = prepare_dataset(test_filepaths, test_labels, is_training=False)

    print("Treinando modelo final...")
    final_model = create_model_from_selection(number_CNN)
    
    final_model.fit(final_train_ds, epochs=EPOCHS, verbose=1)
    
    print("Avaliando no Teste...")
    loss, acc = final_model.evaluate(test_ds)
    
    print(f"\nAcurácia Final (Teste): {acc * 100:.2f}%")
    
    if acc > best_kfold_score:
        print("SUPEROU O K-FOLD! Salvando...")
        final_model.save(f"models/FINAL_BEST_{name_model}_{int(time.time())}.keras")
    else:
        print("Não superou o melhor do K-Fold. Não salvo.")