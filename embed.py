import gc
import os
from time import sleep
import tqdm
import glob
import numpy as np
import torch

from utils import make_directory, split_save_npy, load_txt_or_bpe

# Location of folder containing training .bpe files
TRAIN_PATH = 'Data/Train/'
TEST_PATH = 'Data/Test/'
DEV_PATH = 'Data/Dev/'

from sentence_transformers import SentenceTransformer

model_out_name = 'Out_Roberta'
model = SentenceTransformer('sentence-transformers/all-roberta-large-v1', device="cpu")#.to(device)

def embed_text(text):
    return model.encode(text)

def embed_train_data():
    train_files = glob.glob('{}'.format('{}*'.format(TRAIN_PATH)))

    for file in train_files:
        file_ext_split = os.path.splitext(file)
        data_name = os.path.basename(file_ext_split[0])
        print('\n\nData set: {}'.format(data_name))

        raw_text = load_txt_or_bpe(file)

        embeddings_path = '{}/Train/Embeddings/{}'.format(model_out_name, data_name)
        make_directory(embeddings_path)

        i = 0
        n = len(raw_text)
        done = False
        while not done:
            start = i * 50000
            end = min(start + 50000, n)
            
            file_name = '{}/{}.npy'.format(embeddings_path, i)
            if os.path.exists(file_name):
                i = i + 1
                if end == n:
                    done = True
                continue

            embeddings = model.encode(raw_text[start:end])
            np.save(file_name, embeddings)

            del embeddings
            gc.collect()
        
            i = i + 1
            if end == n:
                done = True
                continue
            
            #sleep(90)

def embed_test_data():
    test_files = glob.glob('{}*'.format(TEST_PATH))
    make_directory('{}/Test/Embeddings'.format(model_out_name))
    for file in test_files:
        file_ext_split = os.path.splitext(file)
        data_name = os.path.basename(file_ext_split[0])
        print('Test data set: {}'.format(data_name))
        embeddings_dir = '{}/Test/Embeddings/{}'.format(model_out_name, data_name)
        make_directory(embeddings_dir)

        test_text = load_txt_or_bpe(file)
        
        test_embeddings = model.encode(test_text)
        split_save_npy(test_embeddings, embeddings_dir)
        print('Saved Embeddings.')

def embed_dev_data():
    dev_files = glob.glob('{}*'.format(DEV_PATH))
    make_directory('{}/Dev/Embeddings'.format(model_out_name))
    for file in dev_files:
        file_ext_split = os.path.splitext(file)
        data_name = os.path.basename(file_ext_split[0])
        print('Dev data set: {}'.format(data_name))
        embeddings_dir = '{}/Dev/Embeddings/{}'.format(model_out_name, data_name)
        make_directory(embeddings_dir)

        dev_text = load_txt_or_bpe(file)
        
        dev_embeddings = model.encode(dev_text)
        split_save_npy(dev_embeddings, embeddings_dir)
        print('Saved Embeddings.')

def embed_full():
    embed_train_data()
    embed_test_data()
    embed_dev_data()
