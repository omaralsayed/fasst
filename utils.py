import numpy as np
import os


PUNC_TO_REMOVE = '!\"\'()*+,./:;<=>?[\]^_`{|}~'

def get_similarity(v1, v2):
    """
    Calculate the cosine/angular similarity between two vectors.

    Args:
        v1 (numpy.ndarray): The first vector.
        v2 (numpy.ndarray): The second vector.

    Returns:
        float: The cosine similarity between v1 and v2.
    """
    numerator = np.dot(v1, v2)
    denominator = np.linalg.norm(v1) * np.linalg.norm(v2)
    similarity = numerator / denominator if denominator != 0 else 0
    return similarity

def make_directory(directory):
    """
    Create a directory on the local machine if it does not exist.

    Args:
        directory (str): The path to the directory to be created.
    """
    os.makedirs(directory, exist_ok=True)

def to_lower_remove_punc(text_list):
    """
    Convert all text in a list to lowercase and remove punctuation.

    Args:
        text_list (list of str): The list of strings to process.

    Returns:
        list of str: The processed list of strings.
    """
    translation_table = str.maketrans('', '', PUNC_TO_REMOVE)
    return [text.lower().translate(translation_table).strip() for text in text_list]
