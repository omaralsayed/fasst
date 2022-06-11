import gc
import os
import numpy as np

from utils import make_directory

import json

DATA_DIR = "./data"

with open("{}/paths.json".format(DATA_DIR), "r") as f:
    paths = json.load(f)

# Delay loading of model until the first time it is used
model = None

def embed_text(text):
    global model
    if model is None:
        try:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer('sentence-transformers/all-roberta-large-v1', device="cpu")
        except ImportError:
            model = None
            print("SentenceTransformer not installed. Install with `pip install sentence-transformers`.")
            raise

    return model.encode(text)


def embed_data(data_set):
    style_names = list(paths[data_set])
    for style in style_names:
        print('Embedding {} data set: {}'.format(data_set, style))

        style_text_path = "{}/".format(DATA_DIR) + paths[data_set][style]["text"]
        with open(style_text_path, "r", errors="ignore") as f:
            style_text = f.read().strip().split('\n')

        style_embedding_path = "{}/".format(DATA_DIR) + paths[data_set][style]["embedding"]

        embeddings = embed_text(style_text)

        make_directory(os.path.dirname(style_embedding_path))
        np.save(style_embedding_path, embeddings)

        del embeddings
        gc.collect()


def embed_full():
    embed_data("test")
    embed_data("train")
    embed_data("dev")
