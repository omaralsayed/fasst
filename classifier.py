import json
import math
import os
import pickle
from collections import Counter

import numpy as np
import tqdm

from utils import (
    get_similarity,
    make_directory,
    to_lower_remove_punc
)


DATA_DIR = "./data"
with open(os.path.join(DATA_DIR, "paths.json"), "r") as f:
    paths = json.load(f)

def create_style_mean_embeddings():
    """
    Create and save mean embeddings for each style based on training data.

    This function calculates the mean embedding vector for each style present in the training dataset.
    It expects a predefined structure in the 'paths' dictionary that includes paths to the embeddings
    for each style and the destination for saving the calculated mean embeddings (centroids).
    
    Requires:
    - Embeddings for each style to be precomputed and saved.
    - The 'paths' dictionary to contain paths for 'train' embeddings and 'centroids'.
    """
    style_names = paths.get("train", {})

    for style in tqdm.tqdm(style_names, desc="Calculating centroids..."):
        train_embeddings_path = os.path.join(DATA_DIR, paths["train"][style]["embedding"])

        if not os.path.exists(train_embeddings_path):
            error_message = f"Embeddings not found at '{train_embeddings_path}'. " \
                            "Cannot continue. Need to first embed text."
            print(error_message)
            raise FileNotFoundError(error_message)

        embeddings = np.load(train_embeddings_path)
        style_vector = np.mean(embeddings, axis=0)
        style_centroid_path = os.path.join(DATA_DIR, paths["centroids"][style]["self"])

        make_directory(os.path.dirname(style_centroid_path))
        np.save(style_centroid_path, style_vector)    

def load_style_mean_embeddings(target_style):
    """
    Load style mean embeddings vectors from files. If they do not exist, create them.

    Args:
        target_style (str): The target style for which to load the mean embeddings.

    Returns:
        dict: A dictionary with style names as keys and their corresponding means.
    """
    self_centroid_path = os.path.join(DATA_DIR, paths["centroids"][target_style]["self"])
    
    # Check if centroid files exist, if not, compute them
    if not os.path.exists(self_centroid_path) or \
       any(not os.path.exists(os.path.join(DATA_DIR, paths["centroids"][target_style]["opposing"][opposing])) 
           for opposing in paths["centroids"][target_style]["opposing"]):
        print('Centroids are not yet computed. Computing...')
        create_style_mean_embeddings()
    
    centroids_dict = {
        target_style: np.load(self_centroid_path)
    }
    
    # Load opposing centroids
    for opposing, path in paths["centroids"][target_style]["opposing"].items():
        opposing_path = os.path.join(DATA_DIR, path)
        if not os.path.exists(opposing_path):
            # This should not happen as create_style_mean_embeddings() was already called
            raise FileNotFoundError(f'Centroid file {opposing_path} not found after creating centroids.')
        centroids_dict[opposing] = np.load(opposing_path)

    return centroids_dict

def idf(word, documents):
    """
    Calculate the inverse document frequency of a word in a collection of documents.

    Args:
        word (str): The word for which to calculate the IDF.
        documents (list of str): The collection of documents.

    Returns:
        float: The IDF of the word.
    """
    n = sum(1 for document in documents if word in document)
    return math.log(len(documents) / n) if n else 0

def tfidf(document, documents):
    """
    Calculate the TF-IDF for all unique words in a document that occur at least 3 times.

    Args:
        document (str): The document for which to calculate TF-IDF scores.
        documents (list of str): The collection of documents to use for IDF calculation.

    Returns:
        dict: A dictionary of words and their scores, sorted by score in descending order.
    """
    tf_dict = Counter(document.split())

    # Only keep words that occur at least 3 times
    tf_dict = {word: freq for word, freq in tf_dict.items() if freq >= 3}

    tf_idf_dict = {
        word: freq * idf(word, documents)
        for word, freq in tqdm.tqdm(tf_dict.items())
    }

    dict_mean = np.mean(list(tf_idf_dict.values()))
    dict_std = np.std(list(tf_idf_dict.values()))

    tf_idf_dict = {
        word: (score - dict_mean) / dict_std
        for word, score in tf_idf_dict.items()
    }

    return dict(sorted(tf_idf_dict.items(), key=lambda x: x[1], reverse=True))

def create_style_tfidf_dictionaries():
    """
    Creates TF-IDF dictionaries for each style and saves them as pickle files.
    If a TF-IDF dictionary file already exists for a style, it is skipped.
    """
    style_names = paths.get("train", {}).keys()

    document_style_dict = {}
    for style in style_names:
        text_file_path = os.path.join(DATA_DIR, paths["train"][style]["text"])
        with open(text_file_path, "r", encoding="utf-8") as file:
            src_text = file.read().splitlines()
        
        processed_text = to_lower_remove_punc(src_text)
        document_style_dict[style] = ' '.join(processed_text)
    
    for style, document in document_style_dict.items():
        pkl_file_path = os.path.join(DATA_DIR, paths["tfidf"][style]["self"])
        if os.path.exists(pkl_file_path):
            print(f"TF-IDF dictionary for style '{style}' already exists. Skipping.")
            continue
        
        # Compute TF-IDF dictionary for the style
        tfidf_dict = tfidf(document, document_style_dict.values())
        
        # Save the TF-IDF dictionary as a pickle
        make_directory(os.path.dirname(pkl_file_path))
        with open(pkl_file_path, 'wb') as pkl_file:
            pickle.dump(tfidf_dict, pkl_file)
        print(f"TF-IDF dictionary for style '{style}' created and saved.")

def load_style_tfidf_dicts(target_style):
    """
    Load TF-IDF dictionaries for a given style and its opposing styles from files.
    If any dictionary does not exist, all necessary dictionaries are created.

    Args:
        target_style (str): The target style for which to load the TF-IDF dictionaries.
    
    Returns:
        dict: A dictionary containing the loaded TF-IDF dictionaries for opposing style.
    """
    self_tfidf_path = os.path.join(DATA_DIR, paths["tfidf"][target_style]["self"])

    # Gather all required TF-IDF dictionary paths
    tfidf_paths = {target_style: self_tfidf_path}
    tfidf_paths.update({
        opposing: os.path.join(DATA_DIR, path)
        for opposing, path in paths["tfidf"][target_style]["opposing"].items()
    })

    # Check if any of the dictionaries are missing
    if any(not os.path.exists(path) for path in tfidf_paths.values()):
        print('TF-IDF Dictionaries are not yet created. Creating...')
        create_style_tfidf_dictionaries()

    tf_idf_dicts = {
        style: pickle.load(open(path, 'rb'))
        for style, path in tfidf_paths.items()
    }

    return tf_idf_dicts

def classify_using_centroids(text_embedding, style_mean_dict):
    """
    Calculate the classification for a single text embedding by choosing the style
    with the highest cosine similarity between the text embedding and style centroids.

    Args:
        text_embedding (array): The embedding of the text to classify.
        style_mean_dict (dict): A dictionary mapping style names to their centroids.

    Returns:
        str: The name of the style with the highest similarity to the text embedding.
    """
    sim_dict_score = {
        style_name: get_similarity(text_embedding, style_mean)
        for style_name, style_mean in style_mean_dict.items()
    }

    return max(sim_dict_score, key=sim_dict_score.get)

def classify_tfidf(text_embedding, text_string, style_mean_dict, style_tfidf_dict, lambda_score):
    """
    Calculate the classification for a single text embedding using a combination of cosine similarity and TF-IDF scores.

    Args:
        text_embedding (array): The embedding of the text to classify.
        text_string (str): The original text string.
        style_mean_dict (dict): A dictionary mapping style names to their centroid embeddings.
        style_tfidf_dict (dict): A dictionary of dictionaries, each mapping words to their TFIDF scores.
        lambda_score (float): The weight given to the TF-IDF score in the overall classification score.

    Returns:
        str: The name of the style with the highest classification score.
    """
    sim_dict_score = {
        style_name: get_similarity(style_mean, text_embedding)
        for style_name, style_mean in style_mean_dict.items()
    }

    style_tfidf_dict_score = {
        style_name: np.mean([tfidf_dict[word] for word in text_string.lower().split() if word in tfidf_dict])
        for style_name, tfidf_dict in style_tfidf_dict.items()
    }

    final_scores = {
        style: (1 - lambda_score) * sim_score + lambda_score * tfidf_score
        for style, sim_score, tfidf_score in zip(style_mean_dict.keys(), sim_dict_score.values(), style_tfidf_dict_score.values())
    }

    return max(final_scores, key=final_scores.get)
