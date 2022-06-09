import glob
import numpy as np
import os
import pickle

import tqdm

from utils import get_similarity, make_directory

import json

with open('./data/paths.json') as f:
    paths = json.load(f)

DATA_DIR = "./data"
# model_out_name = 'Out_Roberta'
embedding_length = 1024

# TRAIN_PATH = 'Data/Train/'


def create_style_mean_embeddings(target_style):
    """Creates mean embeddings for each style. This creates and saves them, then they need to be loaded.
    """
    train_embeddings_path = "/".format(DATA_DIR) + paths["train"][target_style]["embedding"]

    if not os.path.isdir(train_embeddings_path):
        print('Embeddings not found at \'{}\'. Cannot continue. Need to embed text'.format(train_embeddings_path))
        raise Exception('Embeddings not found at \'{}\''.format(train_embeddings_path))
    
    # TODO: Fix the following to be consistent with new structure
    # style_mean_vectors = np.empty((0, embedding_length), np.float32)
    # style_dirs = sorted(glob.glob('{}/*'.format(train_embeddings_path)))
    # style_names = []
    # for style_dir in tqdm.tqdm(style_dirs, desc='Style Mean Vectors'):
    #     style = os.path.basename(style_dir)
    #     style_names.append(style)
    #     style_embeddings_path = '{}/{}'.format(train_embeddings_path, style)
    #     mean_vectors = np.empty((0, embedding_length), np.float32)
    #     embedding_files = sorted(glob.glob('{}/*.npy'.format(style_embeddings_path)))
    #     for file in embedding_files:
    #         embeddings = np.load(file)
    #         mean_vectors = np.vstack((mean_vectors, np.mean(embeddings, axis=0)))
    #     style_mean_vectors = np.vstack((style_mean_vectors, np.mean(mean_vectors, axis=0)))
    
    # train_centroid_path = '{}/Style_Mean_Vectors'.format(model_out_name)
    # make_directory(train_centroid_path)
    # for i in range(len(style_names)):
    #     style_vector = style_mean_vectors[i]
    #     other_styles_mean = np.vstack((style_mean_vectors[0:i],style_mean_vectors[i+1:len(style_names)]))
    #     other_styles_mean = np.mean(other_styles_mean, axis=0)

    #     style_vector = np.vstack((style_vector, other_styles_mean))
    #     np.save('{}/{}.npy'.format(train_centroid_path, style_names[i]), style_vector)


# TODO: Make this work for current codebase
# def create_style_tfidf_dictionaries():
#     # Load text from all styles
#     train_files = sorted(glob.glob('{}*'.format(TRAIN_PATH)))
    
#     # Make directory
#     out_dir = 'Out/Style_TFIDF_Dictionaries'
#     make_directory(out_dir)

#     print('Loading training text...')
#     document_style_dict = dict() # style : [all_text]
#     for file in train_files:
#         file_ext_split = os.path.splitext(file)
#         data_name = os.path.basename(file_ext_split[0])
#         print('\n\nData set: {}'.format(data_name))

#         # Load data
#         raw_text = load_txt_or_bpe(file)
        
#         raw_text = to_lower_remove_punc(raw_text)
#         document_style_dict[data_name] = ' '.join(raw_text)
    
#     for data_name, document in document_style_dict.items():
#         pkl_name = '{}/{}.pkl'.format(out_dir, data_name)
#         if os.path.exists(pkl_name):
#             continue

#         print('\n\nData set: {}'.format(data_name))
#         tfidf_dict = tfidf(document, document_style_dict.values())
#         with open(pkl_name, 'wb') as f:
#             pickle.dump(tfidf_dict, f)


def load_style_mean_embeddings(target_style):
    """Load style mean embeddings vectors from files, or create them if they do not exist.
    """
    train_centroid_path = "{}/centroids".format(DATA_DIR)
    if not os.path.isdir(train_centroid_path):
        print('Centroids are not yet computed. Computing...')
        create_style_mean_embeddings()

    centroids_dict = dict()
    files = sorted(glob.glob('{}/*.npy'.format(train_centroid_path)))
    for file in files:
        centroids_dict[os.path.basename(file).replace(".npy", "")] = np.load(file)

    return centroids_dict


def load_style_tfidf_dict():
    """Load TFIDF dictionaries from files, or create them if they do not exist.
    """
    dicts_path = "{}/tfidf".format(DATA_DIR)

    if not os.path.exists(dicts_path):
        print('TFIDF Dictionaries are not yet created. Creating...')
        # create_style_tfidf_dictionaries() # TODO: Uncomment when func is done
    
    tf_idf_dicts = dict()
    files = sorted(glob.glob('{}/*.pkl'.format(dicts_path)))
    for file in files:
        with open(file, 'rb') as f:
            tf_idf_dicts[os.path.basename(file).replace('.pkl', '')] = pickle.load(f)

    return tf_idf_dicts


def classify_using_centroids(text_embedding, style_mean_dict):
    """Calculates the classification for a single text embedding.
    """
    sim_dict_score = dict()
    for style_name, style_mean in style_mean_dict.items():
        # First, calculate cosine similarity between centroids and text embedding
        centroid_embedding = style_mean[0]
        sim_dict_score[style_name] = get_similarity(text_embedding, centroid_embedding)

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
        style_mean_embedding = style_mean[0]
        sim_dict_score[style_name] = get_similarity(style_mean_embedding, text_embedding)

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