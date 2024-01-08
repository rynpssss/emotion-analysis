import os

from transformers import AutoTokenizer

# Python実行用
MODEL_PATH = './models/model/'
TOKEN_PATH = './models/token/'
STANDARD_SCALER_PATH = './models/pca/standard_scaler.pkl'
PCA_PATH = './models/pca/pca.pkl'
MINMAX_SCALER_PATH = './models/pca/minmax_scaler.pkl'

# Notebook実行用
MODEL_PATH_NOTEBOOK = MODEL_PATH.replace('./', '../')
TOKEN_PATH_NOTEBOOK = TOKEN_PATH.replace('./', '../')
STANDARD_SCALER_PATH_NOTEBOOK = STANDARD_SCALER_PATH.replace('./', '../')
PCA_PATH_NOTEBOOK = PCA_PATH.replace('./', '../')
MINMAX_SCALER_PATH_NOTEBOOK = MINMAX_SCALER_PATH.replace('./', '../')


def read_model(dir):
    if os.path.isdir(dir):
        model = AutoTokenizer.from_pretrained(dir)
        return model
    else:
        print('No directory')
        return None