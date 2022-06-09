import glob
import numpy as np
import os
import tqdm

# Model is only loaded when it is used.
mpnet_base_v2 = None

roberta = None

def embed_text_model(text, model):
    """Embeds a text using a text embedding model.

        Args:
            text (str): Text to embed.
            model (SentenceTransformer): Text embedding model.

        Returns:
            numpy.ndarray: Embedded text.
    """
    if model == None:
        raise Exception('embed_text_model: model is invalid.')

    return model.encode(text)

def embed_text(text):
    """Embeds a text using mpnet_base_v2.

        Args:
            text (str): Text to embed.

        Returns:
            numpy.ndarray: Embedded text.
    """
    # If this is first function call, model is not yet loaded.
    global mpnet_base_v2
    if mpnet_base_v2 is None:
        print('Loading mpnet_base_v2 model.')
        try:
            from sentence_transformers import SentenceTransformer
            mpnet_base_v2 = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
        except ImportError:
            mpnet_base_v2 = None
            print('SentenceTransformer not installed. Install with `pip install sentence-transformers`')
            raise

    return embed_text_model(text, mpnet_base_v2)

def decode_bpe_file(file_path, use_tqdm=True):
    """Decodes a .bpe file to text.

        Args:
            file (str): Path to BPE file.
            roberta (torch.nn.Module): RoBERTa model.
        
        Returns:
            list: Decoded text.
    """
    # If this is first function call, roberta is not yet loaded.
    global roberta
    if roberta is None:
        print('Loading RoBERTa model.')
        try:
            import torch
            roberta = torch.hub.load('pytorch/fairseq', 'roberta.base')
        except:
            print('Torch not installed. Install with `pip install torch`')
            raise
    
    if roberta is None:
        print('RoBERTa model is not properly loaded.')
        return []

    with open(file_path, "r") as f:
        data_bpe = f.read().strip().split('\n')

    data = [roberta.bpe.decode(x) for x in (tqdm.tqdm(data_bpe, desc='Decoding .bpe') if use_tqdm else data_bpe)]
    return data

def get_similarity(v1, v2):
    """Calculates the similarity between two text embeddings.

        Args:
            v1 (numpy.ndarray): First text embedding.
            v2 (numpy.ndarray): Second text embedding.

        Returns:
            float: Similarity between the two embeddings.
    """
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

def get_information_loss(v1, v2):
    """Calculates the information loss between two text embeddings.

        Args:
            v1 (numpy.ndarray): First text embedding.
            v2 (numpy.ndarray): Second text embedding.

        Returns:
            float: Information loss between the two embeddings.
    """
    return 1 - get_similarity(v1, v2)

def get_coherence_score(d1):
    """Calculates the coherence score of a document.

        Args:
            d1 (numpy.ndarray): Document.

        Returns:
            float: Coherence score of the document.
    """
    words = d1.split() # Words in the document
    score = np.array([]).astype(np.float32)

    for index, word in enumerate(words):
        # Get the embedding of the current word
        v1 = embed_text(word, embedding_model)

        # Get the embedding of the next word
        if index < len(words) - 1:
            v2 = embed_text(words[index + 1], embedding_model)
        else:
            v2 = embed_text(words[0], embedding_model)

        score = np.append(score, get_similarity(v1, v2))

    return np.mean(score)

def get_most_cardinal(document, k):
    """Calculates the k most cardinal words in a document.

        Args:
            document (str): Document.
            k (int): Number of words to return.

        Returns:
            list: k most cardinal words in the document.
    """
    words = document.split() # Words in the document
    cards = dict() # Dictionary of cardinality of each word

    for word in words:
        if word in cards:
            cards[word] += 1
        else:
            cards[word] = 1
    
    # Sort the dictionary by cardinality
    sorted_cards = sorted(cards.items(), key=lambda x: x[1], reverse=True)

    # Return the k most cardinal words
    return [card[0] for card in sorted_cards[:k]]

def download_glove_embeddings(path="glove.42B.300d"):
    """Downloads the GloVe embeddings.

        Args:
            path (str): Path to save the embeddings.
    """
    import zipfile
    import urllib.request

    # If folder already exists, return True
    if os.path.isdir(path):
        print('GloVe embeddings already downloaded.')
        return True

    # Download the GloVe embeddings
    urllib.request.urlretrieve("https://nlp.stanford.edu/data/glove.42B.300d.zip", "glove.42B.300d.zip")

    # Extract the embeddings from Zip
    with zipfile.ZipFile(path + ".zip", 'r') as zip_ref:
        zip_ref.extractall(path)

    # Delete the archive
    os.remove("glove.42B.300d.zip")
    return False

def load_glove_embeddings(path="glove.42B.300d"):
    """Loads the GloVe embeddings.

        Args:
            path (str): Path to the embeddings.

        Returns:
            dict: Dictionary of embeddings.
    """
    import json

    try:
        with open(path + "/glove.42B.300d.txt", "rb") as f:
            embeddings = json.load(f)
    except FileNotFoundError:
        print('Downloading GloVe embeddings...')
        download_glove_embeddings(path)
        with open(path + "/glove.42B.300d.txt", "rb") as f:
            embeddings = json.load(f)

    # Return the embeddings
    return embeddings

def make_directory(directory):
    """Attempts to create a directory if it does not already exist.

        Args:
            directory (str): Path and name of directory to create.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)

def save_text_file(text, file_path):
    """Save text to a file.

        Args:
            text (str): Text to save.
            file_path (str): Path to save at.
    """
    with open(file_path, 'w') as f:
        for item in text:
            f.write("%s\n" % item)

# Number of items per out file, so we don't have 15 gb files
FILE_SPLIT = 50000
def split_save_txt(data, folder_path, file_split=FILE_SPLIT):
    """Split data into portions of file_split and save it to separate .txt files.

        Args:
            data ([string]): List of data to split and save.
            folder_path (str): Path of directory to save split data into.
            file_split (int): Number of items per split, default to FILE_SPLIT.
    """
    i = 0
    n = len(data)
    done = False
    while not done:
        start = i * file_split
        end = min(start + file_split, n)
        
        # Saving bigrams as .txt uses significantly less space
        save_text_file(data[start:end], '{}/{}.txt'.format(folder_path, i))
    
        i = i + 1
        if end == n:
            done = True

def split_save_npy(data, folder_path, file_split=FILE_SPLIT):
    """Split data into portions of FILE_SPLIT and save it to separate .npy files.

        Args:
            data (numpy.ndarray): Array of data to split and save.
            folder_path (str): Path of directory to save split data into.
            file_split (int): Number of items per split, default to FILE_SPLIT.
    """
    i = 0
    n = len(data)
    done = False
    while not done:
        start = i * file_split
        end = min(start + file_split, n)
        
        np.save('{}/{}.npy'.format(folder_path, i), data[start:end])
    
        i = i + 1
        if end == n:
            done = True

def load_split_txt(folder_path, file_split=FILE_SPLIT):
    """Loads split .txt files.

        Args:
            folder_path (str): Path of directory to load split data from.
            file_split (int): Number of items per split, default to FILE_SPLIT.

        Returns:
            list: List of data from split files.
    """
    data = []
    for file in glob.glob('{}/*.txt'.format(folder_path)):
        with open(file, 'r') as f:
            data.extend(f.read().splitlines())
    
    return data

# TODO: Make a version of this that returns an iterable and loads file_split at a time
def load_split_npy(folder_path, file_split=FILE_SPLIT):
    """Loads split .npy files.

        Args:
            folder_path (str): Path of directory to load split data from.
            file_split (int): Number of items per split, default to FILE_SPLIT.

        Returns:
            numpy.ndarray: Array of data from split files.
    """
    data = []

    for file in glob.glob('{}/*.npy'.format(folder_path)):
        if len(data) == 0:
            data = np.load(file)
        else:
            temp_d = np.load(file)
            data = np.vstack((data,temp_d))
    
    return data

def load_txt_or_bpe(file_path):
    """Loads a .txt file if it exists, otherwise loads a .bpe file.

        Args:
            file_path (str): Path of file to load.

        Returns:
            list: List of data from file.
    """
    file_ext_split = os.path.splitext(file_path)
    
    # Load data
    if file_ext_split[1] == '.bpe':
        raw_text = decode_bpe_file(file_path)
    else:
        with open(file_path, "r", errors="ignore") as f:
            raw_text = f.read().strip().split('\n')
    
    return raw_text

PUNC_TO_REMOVE = '!\"\'()*+,./:;<=>?[\]^_`{|}~'
def to_lower_remove_punc(text_list):
    """Converts text to lowercase and removes punctuation.

        Args:
            text_list ([string]): List of text to processed.
        Returns:
            [string]: List of processed text.
    """
    return [line.translate(str.maketrans('', '', PUNC_TO_REMOVE)).strip().lower() for line in text_list]