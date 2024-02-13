"""    
1. Compute vector spaces for the source and target texts using RoBERTa-Large to obtain embeddings.

2. Align the styles of source and target by adjusting the embeddings:
    a. Calculate the mean vector of the source and target embeddings.
    b. Shift the mean of the source embeddings towards the target embeddings to align the styles.

3. Compute the similarity between the source (src) and target (tgt) embeddings:
    a. Use cosine similarity or another appropriate metric to measure similarity between source and target.

4. Extract the k-nearest neighbors (k-NN) of the target embeddings based on the computed similarity scores:
    a. For each source embedding, identify the k most similar target embeddings.
    b. These form the k-nearest neighbors for each source text in the target style space.

5. Summarize the aligned text using an OPT model for text summarization:
    a. Input the k-nearest neighbors into the OPT model with the proper prompt.
    b. Generate summaries that incorporate the target style characteristics.
"""

import argparse
import gc
import json
import os
import re
from itertools import combinations

import numpy as np
import torch
import tqdm
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, pipeline

from metrics import evaluate
from utils import get_similarity


device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Running on {device}")

model = SentenceTransformer('sentence-transformers/all-roberta-large-v1')

DATA_DIR   = "./data"
OUTPUT_DIR = "./output"

OPT_MODEL = "facebook/opt-1.3b"

with open("{}/prompts.json".format(DATA_DIR), "r") as f:
    prompts = json.load(f)

with open("{}/paths.json".format(DATA_DIR), "r") as f:
    paths = json.load(f)

generator = pipeline('text-generation', model=OPT_MODEL, device=0
                     if device == "cuda" else -1)
tokenizer = AutoTokenizer.from_pretrained(OPT_MODEL, use_fast=False)

def clean():
    """
    Perform garbage collection and clear CUDA cache if CUDA is available.
    """
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache() 

def preprocess(text):
    """
    Preprocesses the given text for further processing or model input.

    Args:
        text (str): The text to preprocess.

    Returns:
        str: The preprocessed/cleaned text.
    """
    text = text.replace("\n", " ")
    return re.sub(r"\s+(?=[.,;:!?])", "", text).strip()

def subset(arr, r):
    """
    Generate all possible subsets of size r from the array.

    Args:
        arr (list): The input list from which subsets are generated.
        r (int): The size of each subset to be generated.

    Returns:
        list of lists: A list containing all subsets of given size r.
    """
    return [list(tup) for tup in combinations(arr, r)]

def get_neighbors(src_vector, tgt_vectors, k):
    """
    Find the indices of the k most similar target vectors to a source vector.

    Args:
        src_vector (array-like): The source vector from which similarity is measured.
        tgt_vectors (array-like): A collection of target vectors to compare against the source vector.
        k (int): The number of nearest neighbors to return.

    Returns:
        np.ndarray: An array of indices corresponding to the k most similar vectors in tgt_vectors.
    """
    similarities = np.array([get_similarity(src_vector, tgt_vector) for tgt_vector in tgt_vectors])
    return np.argsort(similarities)[::-1][:k]

def aligned_embed(args, alignment_strength=1):
    """
    Aligns the input embeddings with the target style embeddings by adjusting their vector spaces.

    This function loads the input, source, and target embeddings, computes their means,
    and aligns the source and target embedding spaces based on global mean and specified strength.

    Args:
        args (Namespace): A namespace object containing source_style and target_style attributes.
        alignment_strength (float, optional): The strength of alignment. Defaults to 1.

    Returns:
        tuple: A tuple containing aligned source embeddings and aligned target embeddings.
    """
    print("Loading Input Embeddings...")
    input_embedding_path = os.path.join(DATA_DIR, paths["test"][args.source_style]["embedding"])
    input_embedding = np.load(input_embedding_path)

    print("Loading Train Embeddings...")
    source_embedding_path = os.path.join(DATA_DIR, paths["train"][args.source_style]["embedding"])
    target_embedding_path = os.path.join(DATA_DIR, paths["train"][args.target_style]["embedding"])
    source_embedding = np.load(source_embedding_path)
    target_embedding = np.load(target_embedding_path)

    source_mean = np.mean(source_embedding, axis=0)
    target_mean = np.mean(target_embedding, axis=0)
    global_mean = np.mean(np.concatenate([source_embedding, target_embedding], axis=0), axis=0)

    print("Aligning Vector Spaces...")
    aligned_src = input_embedding + alignment_strength * (global_mean - source_mean)
    aligned_tgt = target_embedding + alignment_strength * (global_mean - target_mean)

    return aligned_src, aligned_tgt

def summarize(args, text):
    """
    Generates a summary for a given text, styled according to the target style specified in args.

    Args:
        args (Namespace): Argument namespace containing target_style.
        text (str): Text to summarize.

    Returns:
        str: Styled summary of the input text.
    """
    prompt = prompts["examples"][args.target_style]
    context = f"{prompt} {text}\n"

    # Encode the context text data
    input_ids = tokenizer(context, return_tensors="pt").input_ids.to(device)

    # Generate the styled summary
    output = generator(context, max_length=input_ids.shape[1] + 32)[0]["generated_text"]
    output = output.removeprefix(context)

    return output.replace(prompts["prefixes"][args.target_style], "").split("\n")[0].strip()

def run(args):
    """
    Process text by extracting k-nearest neighbors, generating candidates,
    and evaluating to select the best option based on alignment strength.

    Args:
        args (Namespace): Argument namespace containing target_style.
    """
    k, r = args.k, args.r

    source_text_path = os.path.join(DATA_DIR, paths["test"][args.source_style]["text"])
    target_text_path = os.path.join(DATA_DIR, paths["train"][args.target_style]["text"])
    
    with open(source_text_path, "r", encoding="utf-8") as f:
        src_text = f.readlines()
    with open(target_text_path, "r", encoding="utf-8") as f:
        tgt_text = f.readlines()

    input_file = os.path.join(OUTPUT_DIR, f"{args.source_style}_to_{args.target_style}_input_k{k}_r{r}.txt")
    output_file = os.path.join(OUTPUT_DIR, f"{args.source_style}_to_{args.target_style}_output_k{k}_r{r}.txt")

    open(input_file, "w").close()
    open(output_file, "w").close()

    aligned_src, aligned_tgt = aligned_embed(args, alignment_strength=1)

    print("Extracting k-nearest neighbors...")
    candidates = [get_neighbors(src_vector, aligned_tgt, k) for src_vector in tqdm.tqdm(aligned_src)]

    comb = {i: subset(v, r) for i, v in enumerate(candidates)}

    print("Generating... Check output file...")
    for l, v in comb.items():
        # Process and summarize neighbor texts
        src_neighbor_text = [" ".join([preprocess(tgt_text[i]) for i in neighbor_indices]) for neighbor_indices in v]

        evaluations = [(summarize(args, text), *evaluate(args.target_style, [src_text[l]], [text], float(args.lambda_score)))
                       for text in src_neighbor_text]

        # Select the best option (score-based)
        scores = [acc * cos_sim * cola_acc for _, acc, cos_sim, cola_acc in evaluations]
        fallback = [(acc + cos_sim + cola_acc) / 3 for _, acc, cos_sim, cola_acc in evaluations]
        best_option = evaluations[np.argmax(scores if np.sum(scores) != 0 else fallback)][0]

        with open(input_file, "a") as f_input, open(output_file, "a") as f_output:
            f_input.write(src_text[l])
            f_output.write(f"{best_option}\n")

        clean()

def main():
    parser = argparse.ArgumentParser(description="Process text data with specified parameters.")
    parser.add_argument("--k", type=int, default=4, help="Number of nearest neighbors to consider.")
    parser.add_argument("--r", type=int, default=3, help="Subset size for summarization.")
    parser.add_argument("--input", type=str, default="", help="Path to the input file.")
    parser.add_argument("--source_style", type=str, default="", help="Style of the source text.")
    parser.add_argument("--target_style", type=str, default="", help="Desired target style for the output text.")
    parser.add_argument("--lambda_score", type=str, default="", help="Lambda score for evaluation.")

    args = parser.parse_args()
    run(args)

if __name__ == "__main__":
    main()
