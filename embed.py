import gc
import json
import os

import numpy as np

from utils import make_directory


DATA_DIR = "./data"
with open(os.path.join(DATA_DIR, "paths.json"), "r") as f:
    paths = json.load(f)

model = None  # Delay loading of model until the first time it is used

def embed_text(text):
    """
    Embeds a given text using a pre-trained sentence transformer model.
    
    Args:
        text (list of str): Text to be embedded.
    
    Returns:
        numpy.ndarray: The embeddings of the input text.
    """
    global model
    if model is None:
        try:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer('sentence-transformers/all-roberta-large-v1')
        except ImportError as e:
            print("Error: SentenceTransformer not installed. Please install with `pip install sentence-transformers`.")
            raise e

    return model.encode(text)

def embed_data(data_set):
    """
    Embeds text data for a given dataset and style, saving the embeddings to a file.
    
    Args:
        data_set (str): The dataset to be processed (e.g., 'test', 'train', 'dev').
    """
    for style, paths_info in paths[data_set].items():
        print(f'Embedding {data_set} data set: {style}')

        style_text_path = os.path.join(DATA_DIR, paths_info["text"])
        with open(style_text_path, "r", errors="ignore") as f:
            style_text = f.read().strip().split('\n')

        embeddings = embed_text(style_text)

        style_embedding_path = os.path.join(DATA_DIR, paths_info["embedding"])
        make_directory(os.path.dirname(style_embedding_path))
        np.save(style_embedding_path, embeddings)

        del embeddings
        gc.collect()

def embed_full():
    """
    Embeds text data for 'test', 'train', and 'dev' datasets.
    """
    for data_set in ["test", "train", "dev"]:
        embed_data(data_set)

if __name__ == "__main__":
    embed_full()
