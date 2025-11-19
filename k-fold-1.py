import os
import pandas as pd
import numpy as np
import tensorflow as tf
import time
import gc
from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint
from tensorflow.keras import mixed_precision
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2, VGG19, VGG16
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess
from tensorflow.keras.applications.vgg19 import preprocess_input as vgg19_preprocess
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg16_preprocess
from tensorflow.keras import models, layers
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import StratifiedKFold
import sys

policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)

list_CNNs = ['MobileNetV2', 'VGG19', 'VGG16']

i = 1
for CNN in list_CNNs:
    print(f"Digite {i} para usar {CNN}")
    i += 1
number_CNN = int(input("Escolha o modelo (1, 2, ou 3): "))
name_model = "" 


if number_CNN == 1:
    preprocessing_function = mobilenet_preprocess
    name_model = "MobileNetV2"
elif number_CNN == 2:
    preprocessing_function = vgg19_preprocess 
    name_model = "VGG19"
elif number_CNN == 3:
    preprocessing_function = vgg16_preprocess 
    name_model = "VGG16"
else:
    raise ValueError(f"Modelo desconhecido: {number_CNN}")

print(f"Modelo selecionado: {name_model}")


train_datagen = ImageDataGenerator(
    preprocessing_function=preprocessing_function, 
    rotation_range=40,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    brightness_range=(0.7, 1.3),
    shear_range=0.2,
    fill_mode='nearest' 
)

val_datagen = ImageDataGenerator(
    preprocessing_function=preprocessing_function 
)


BASE_DIR = os.path.expanduser("~/UNION FOLDS")

# Define os caminhos CORRETOS (Pool para Pool, Teste para Teste)
POOL_DIR = os.path.join(BASE_DIR, "TRAIN_VAL_POOL")
TEST_DIR = os.path.join(BASE_DIR, "TEST_SET")


def verificar_diretorios():
    diretorios_necessarios = ['models', 'logs']
    for dir_name in diretorios_necessarios:
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
            print(f"Diretório {dir_name} criado.")
            
verificar_diretorios() 

CLASSES = ["HBS", "ALL", "AML"]
TARGET_SIZE = (224, 224)
BATCH_SIZE = 64
EPOCHS = 50
LEARNING_RATE = 0.0002
DROPOUT_RATE = 0.2                                  
K_FOLDS = 5

print(f"Carregando caminhos de arquivos de : {POOL_DIR}")
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
                all_labels.append(class_name)
    except FileNotFoundError:
            print(f"[AVISO] Pasta de classe não encontrada: {class_dir}. Pulando")

if not all_filepaths:
    print(f"[ERRO FATAL] Nenhum arquivo de imagem encontrado em {POOL_DIR}")
    sys.exit()

x_pool = np.array(all_filepaths)
y_pool = np.array(all_labels)
print(f"Total de {len(x_pool)} imagens carregadas no 'POOL' para K-FOLD ")



# Chamar dentro do loop K-FOLD passando o modelo escolhido no incio
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
        loss='categorical_crossentropy', 
        metrics=['accuracy']
    )
    return model


print(f"\nIniciando Validação Cruzada K-Fold (K={K_FOLDS}) para o modelo: {name_model}...")

skf = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=42)
fold_scores = []
best_fold_model_paths = []
fold_number = 1

for train_indices, val_indices in skf.split(x_pool, y_pool):
    print("\n" + "="*30)
    print(f"INICIANDO FOLD {fold_number}/{K_FOLDS}")
    print("=" * 30)

    # Pega os caminhos e os rotulos para este fold
    x_train_fold, y_train_fold = x_pool[train_indices], y_pool[train_indices]
    x_val_fold, y_val_fold = x_pool[val_indices], y_pool[val_indices]

    df_train = pd.DataFrame({'filepath': x_train_fold, 'class': y_train_fold})
    df_val = pd.DataFrame({'filepath': x_val_fold, 'class': y_val_fold})

    train_generator = train_datagen.flow_from_dataframe(
        dataframe=df_train, x_col='filepath', y_col='class',
        target_size=TARGET_SIZE, batch_size=BATCH_SIZE,
        class_mode='categorical', classes=CLASSES
    )
    val_generator = val_datagen.flow_from_dataframe(
        dataframe=df_val, x_col='filepath', y_col='class',
        target_size=TARGET_SIZE, batch_size=BATCH_SIZE,
        class_mode='categorical', classes=CLASSES, shuffle=False
    )

    # Passamos o modelo Escolhido no Inicio
    model = create_model_from_selection(number_CNN)

    print(f"Treinando em {len(df_train)} imagens, validando em {len(df_val)} imagens...")

    timestamp = int(time.time())
    caminho_log = f"logs/treino_model{name_model}_bs{BATCH_SIZE}_lr{LEARNING_RATE}_fold{fold_number}_{timestamp}.csv"
    csv_logger = CSVLogger(caminho_log)
    
    caminho_modelo_fold = f"models/BEST_MODEL_{name_model}_bs{BATCH_SIZE}_lr{LEARNING_RATE}_fold{fold_number}_{timestamp}.keras"
 
    # monitora o 'val_accuracy' e salva apenas o melhor 
    checkpoint_callback = ModelCheckpoint(
        filepath=caminho_modelo_fold,
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    )

    history = model.fit(
        train_generator,
        epochs=EPOCHS,             
        validation_data=val_generator, 
        verbose=1,
        callbacks=[csv_logger, checkpoint_callback],

    )
    
    print(f"Avaliando Fold {fold_number}...")
    loss, accuracy = model.evaluate(val_generator)

    print(f"Acurácia do Fold {fold_number}: {accuracy * 100:.2f}%")
    best_fold_accuracy = np.max(history.history['val_accuracy'])
    fold_scores.append(best_fold_accuracy)
    best_fold_model_paths.append(caminho_modelo_fold)
    
    del model
    tf.keras.backend.clear_session()
    gc.collect()
    
    fold_number += 1

print("\n" + "=" * 30)
print(f"TREINAMENTO K-FOLD CONCLUÍDO ({name_model})")
print("="*30)
print(f"Melhor Acurácia de cada um dos {K_FOLDS} folds:")
for i, score in enumerate(fold_scores):
    print(f"  Fold {i+1}: {score * 100:.2f}% (salvo em {best_fold_model_paths[i]})")

if fold_scores:
    best_kfold_score = np.max(fold_scores)
    best_kfold_indice = np.argmax(fold_scores)
    print(f"\nAcurácia Média (K-Fold): {np.mean(fold_scores)*100:.2f}%")
    print(f"Desvio Padrão (K-Fold): {np.std(fold_scores) * 100:.2f}%")
    print(f"MELHOR ACCURACIA K-FOLD: {best_kfold_score*100:.2f}% do fold {best_kfold_indice + 1}")
else:
    print("\nNenhum score foi registrado.")
    best_kfold_score = 0


print("\n" + "=" * 40)
print(f"--- INICIANDO A PROVA FINAL ({name_model}) ---")
print("=" * 40)

print(f"Gerador de treino com todos os {len(x_pool)} arquivos do POOL...")
final_train_generator = train_datagen.flow_from_directory(
    directory=POOL_DIR, 
    target_size=TARGET_SIZE, batch_size=BATCH_SIZE,
    class_mode='categorical', classes=CLASSES, shuffle=True 
)

print(f"Gerador de teste com arquivos do TEST_SET...")
test_generator = val_datagen.flow_from_directory(
    directory=TEST_DIR, 
    target_size=TARGET_SIZE, batch_size=BATCH_SIZE, 
    class_mode='categorical', classes=CLASSES, shuffle=False 
)

print("\nCriando e treinando o modelo final...")

final_model = create_model_from_selection(number_CNN) 

timestamp = int(time.time())
caminho_log_final = f"logs/treino_FINAL_{name_model}_bs{BATCH_SIZE}_lr{LEARNING_RATE}_{timestamp}.csv"
csv_logger_final = CSVLogger(caminho_log_final)

history_final = final_model.fit(
    final_train_generator,
    epochs=EPOCHS, 
    verbose=1,
    callbacks=[csv_logger_final],
)

test_loss, test_accuracy = final_model.evaluate(test_generator)

print("\n---- RESULTADO DA PROVA FINAL ---")
print(f"Perda (loss) no Teste: {test_loss:.4f}")
print(f"Acurácia no Teste: {test_accuracy * 100:.2f}%")

if (test_accuracy > best_kfold_score):
    print(f"\n[SUCESSO] accuracia do teste final ({test_accuracy*100:.2f}%) é MELHOR que a melhor do K-FOLD ({best_kfold_score*100:.2f}%).")
    caminho_modelo_final = f"models/FINAL_BEST{name_model}_bs{BATCH_SIZE}_lr{LEARNING_RATE}_{timestamp}.keras"
    final_model.save(caminho_modelo_final)
    print(f"Modelo FINAL salvo em: {caminho_modelo_final}")

else:
    print(f"\n[INFO] accuracia do teste final ({test_accuracy*100:.2f}%) NÃO é melhor que a do melhor K-FOLD ({best_kfold_score*100:.2f}%).")
    print("Modelo Final não foi salvo")