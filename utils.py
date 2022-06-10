import numpy as np
import os

def get_similarity(v1, v2):
    """Calculates the similarity between two text embeddings.

        Args:
            v1 (numpy.ndarray): First text embedding.
            v2 (numpy.ndarray): Second text embedding.

        Returns:
            float: Similarity between the two embeddings.
    """
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))


def make_directory(directory):
    """Attempts to create a directory if it does not already exist.

        Args:
            directory (str): Path and name of directory to create.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)


PUNC_TO_REMOVE = '!\"\'()*+,./:;<=>?[\]^_`{|}~'
def to_lower_remove_punc(text_list):
    """Converts text to lowercase and removes punctuation.

        Args:
            text_list ([string]): List of text to processed.
        Returns:
            [string]: List of processed text.
    """
    return [line.translate(str.maketrans('', '', PUNC_TO_REMOVE)).strip().lower() for line in text_list]