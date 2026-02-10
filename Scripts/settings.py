import os
import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_WORKERS = 2
PIN_MEMORY = True if torch.cuda.is_available() else False

BASE_DIR = os.path.expanduser("~/UNION FOLDS")
POOL_DIR = os.path.join(BASE_DIR, "TRAIN_VAL_POOL")
TESTE_DIR = os.path.join(BASE_DIR, "TEST_SET")

CLASSES = ["ALL", "AML", "HBS"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
NUM_CLASSES = len(CLASSES)

BATCH_SIZE = 16
LEARNING_RATE = 0.0002
EPOCHS = 30
K_FOLDS = 5

SEED = 42

