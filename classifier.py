import math
import numpy as np
import os
import pickle

import tqdm

from utils import get_similarity, make_directory, to_lower_remove_punc

import json

DATA_DIR = "./data"

with open("{}/paths.json".format(DATA_DIR), "r") as f:
    paths = json.load(f)


def create_style_mean_embeddings():
    """Creates mean embeddings for each style. This creates and saves them, then they need to be loaded.
    """
    style_names = list(paths["train"])
    for style in tqdm.tqdm(style_names, desc="Style Mean Vectors"):
        train_embeddings_path = "{}/".format(DATA_DIR) + paths["train"][style]["embedding"]

        if not os.path.exists(train_embeddings_path):
            print('Embeddings not found at \'{}\'. Cannot continue. Need to embed text'.format(train_embeddings_path))
            raise Exception('Embeddings not found at \'{}\''.format(train_embeddings_path))

        embeddings = np.load(train_embeddings_path)
        style_vector = np.mean(embeddings, axis=0)
        style_centroid_path = "{}/".format(DATA_DIR) + paths["centroids"][style]["self"]
        make_directory(os.path.dirname(style_centroid_path))
        np.save(style_centroid_path, style_vector)       


def load_style_mean_embeddings(target_style):
    """Load style mean embeddings vectors from files, or create them if they do not exist.
    """
    self_centroid_path = "{}/".format(DATA_DIR) + paths["centroids"][target_style]["self"]
    if not os.path.exists(self_centroid_path):
        print('Centroids are not yet computed. Computing...')
        create_style_mean_embeddings()

    opposing_centroids = list(paths["centroids"][target_style]["opposing"])
    for opposing in opposing_centroids:
        if not os.path.exists("{}/".format(DATA_DIR) + paths["centroids"][target_style]["opposing"][opposing]):
            print('Centroids are not yet computed. Computing...')
            create_style_mean_embeddings()

    centroids_dict = dict()
    centroids_dict[target_style] = np.load(self_centroid_path)
    for opposing in opposing_centroids:
        centroids_dict[opposing] = np.load("{}/".format(DATA_DIR) + paths["centroids"][target_style]["opposing"][opposing])

    return centroids_dict


def idf(word, documents):
    n = 0
    for document in documents:
        if word in document:
            n += 1
    return math.log(len(documents) / n)


def tfidf(document, documents):
    tf_dict = {}
    for word in document.split():
        if word in tf_dict:
            tf_dict[word] += 1
        else:
            tf_dict[word] = 1

    # Only keep words that occur at least 3 times
    tf_dict = {k: v for k, v in tf_dict.items() if v >= 3}

    tf_idf_dict = {}
    for word in tqdm.tqdm(tf_dict):
        tf_idf_dict[word] = tf_dict[word] * idf(word, documents)
    
    dict_mean = np.mean(list(tf_idf_dict.values()))
    dict_std = np.std(list(tf_idf_dict.values()))

    for word in tf_idf_dict.keys():
        tf_idf_dict[word] = ((tf_idf_dict[word] - dict_mean) / dict_std)

    return dict(sorted(tf_idf_dict.items(), key=lambda x: x[1], reverse=True))


def create_style_tfidf_dictionaries():
    # Load text from all styles
    style_names = list(paths["train"])

    #print('Loading training text...')
    document_style_dict = dict() # style : [all_text]
    for style in style_names:
        with open("{}/".format(DATA_DIR) +  paths["train"][style]["text"], "r", encoding="utf-8") as f:
            src_text = f.readlines()
        
        src_text = to_lower_remove_punc(src_text)
        document_style_dict[style] = ' '.join(src_text)
    
    for style, document in document_style_dict.items():
        pkl_name = "{}/".format(DATA_DIR) + paths["tfidf"][style]["self"]
        if os.path.exists(pkl_name):
            continue
        
        #print('\n\nData set: {}'.format(style))
        tfidf_dict = tfidf(document, document_style_dict.values())

        make_directory(os.path.dirname(pkl_name))
        with open(pkl_name, 'wb') as f:
            pickle.dump(tfidf_dict, f)


def load_style_tfidf_dicts(target_style):
    """Load TFIDF dictionaries from files, or create them if they do not exist.
    """
    self_tfidf_path = "{}/".format(DATA_DIR) + paths["tfidf"][target_style]["self"]
    if not os.path.exists(self_tfidf_path):
        print('TFIDF Dictionaries are not yet created. Creating...')
        create_style_tfidf_dictionaries()
    
    opposing_tfidf = list(paths["tfidf"][target_style]["opposing"])
    for opposing in opposing_tfidf:
        if not os.path.exists("{}/".format(DATA_DIR) + paths["tfidf"][target_style]["opposing"][opposing]):
            print('TFIDF Dictionaries are not yet created. Creating...')
            create_style_tfidf_dictionaries()

    tf_idf_dicts = dict()
    with open(self_tfidf_path, 'rb') as f:
        tf_idf_dicts[target_style] = pickle.load(f)
    for opposing in opposing_tfidf:
        with open("{}/".format(DATA_DIR) + paths["tfidf"][target_style]["opposing"][opposing], 'rb') as f:
            tf_idf_dicts[opposing] = pickle.load(f)

    return tf_idf_dicts


def classify_using_centroids(text_embedding, style_mean_dict):
    """Calculates the classification for a single text embedding.
    """
    sim_dict_score = dict()
    for style_name, style_mean in style_mean_dict.items():
        # First, calculate cosine similarity between centroids and text embedding
        sim_dict_score[style_name] = get_similarity(text_embedding, style_mean)

    # Calculates classification score and chooses the style with highest similarity
    max_score = -1
    max_style = ""

    for i in range(len(style_mean_dict)):
        style = list(style_mean_dict.keys())[i]

        score = sim_dict_score[style]
        if score > max_score:
            max_score = score
            max_style = style

    return max_style


def classify_tfidf(text_embedding, text_string, style_mean_dict, style_tfidf_dict, lambda_score):
    """Calculates the classification for a single text embedding.
    """
    sim_dict_score = dict()
    style_tfidf_dict_score = dict()
    for style_name, style_mean in style_mean_dict.items():
        # First, calculate cosine similarity between centroids and text embedding
        sim_dict_score[style_name] = get_similarity(style_mean, text_embedding)

        tfidf_dict = style_tfidf_dict[style_name]
        dict_score = 0
        count = 0
        for word in text_string.lower().split():
            if word in tfidf_dict:
                dict_score += tfidf_dict[word]
                count += 1
        if count > 0:
            dict_score /= count
        style_tfidf_dict_score[style_name] = dict_score
    
    # Calculates classification score and chooses the style with highest score
    max_score = -1
    max_style = ''
    
    for i in range(len(style_mean_dict)):
        style = list(style_mean_dict.keys())[i]

        score = (1 - lambda_score) * sim_dict_score[style] + lambda_score * style_tfidf_dict_score[style]

        if score > max_score:
            max_score = score
            max_style = style
    
    return max_style