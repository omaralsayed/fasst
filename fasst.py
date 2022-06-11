# 1. Compute vector spaces using RoBERTa-L
# 2. Align the styles by shifting the mean
# 3. Compute similarity between src and tgt
# 4. Extract the k-nearest neighbors of tgt
# 5. Pass through OPT for text summarization

import argparse
import json

import tqdm
import torch

import re
import gc

import numpy as np

from itertools import combinations
from metrics import *

device = ("cuda" if torch.cuda.is_available() else "cpu")
print("Running on {}".format(device))

from sentence_transformers import SentenceTransformer

model = SentenceTransformer('sentence-transformers/all-roberta-large-v1')
print("RoBERTa Loaded.")

from transformers import AutoTokenizer
from transformers import pipeline

DATA_DIR   = "./data"
OUTPUT_DIR = "./output"

with open("{}/prompts.json".format(DATA_DIR), "r") as f:
    prompts = json.load(f)

with open("{}/paths.json".format(DATA_DIR),   "r") as f:
    paths = json.load(f)

OPT = "opt-1.3b"

if device == "cuda":
    generator = pipeline('text-generation', model="facebook/{}".format(OPT), device=0)
else:
    generator = pipeline('text-generation', model="facebook/{}".format(OPT))
tokenizer = AutoTokenizer.from_pretrained("facebook/{}".format(OPT), use_fast=False)

def clean():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def subset(arr, r):
    tups = list(combinations(arr, r))
    return [list(tup) for tup in tups]


def aligned_embed(args, alignment_strength=1):
    print("Loading Input Embeddings...")
    input_embedding = np.load("{}/".format(DATA_DIR) + paths["test"][args.source_style]["embedding"])

    print("Loading Train Embeddings...")
    source_embedding = np.load("{}/".format(DATA_DIR) + paths["train"][args.source_style]["embedding"])
    target_embedding = np.load("{}/".format(DATA_DIR) + paths["train"][args.target_style]["embedding"])

    source_mean = np.mean(source_embedding, axis=0)
    target_mean = np.mean(target_embedding, axis=0)

    global_mean = np.mean(np.concatenate((source_embedding, target_embedding), axis=0), axis=0)

    print("Aligning Vector Spaces...")
    aligned_src = input_embedding  + alignment_strength * (global_mean - source_mean)
    aligned_tgt = target_embedding + alignment_strength * (global_mean - target_mean)

    return aligned_src, aligned_tgt


def similarity(src_embeddings, tgt_embeddings):
    return np.dot(src_embeddings, tgt_embeddings) / \
              (np.linalg.norm(src_embeddings) * np.linalg.norm(tgt_embeddings))


def get_neighbors(src_vector, tgt_vectors, k):
    similarities = np.array([similarity(src_vector, tgt_vector) \
        for tgt_vector in tgt_vectors])
    return np.argsort(similarities)[::-1][:k]


def summarize(args, text):
    prompt = prompts["examples"][args.target_style]
    context = prompt + " " + text + "\n"

    if device == "cuda":
        input_ids = tokenizer(context, return_tensors="pt").input_ids.cuda()
    else:
        input_ids = tokenizer(context, return_tensors="pt").input_ids

    output = generator(context, max_length=input_ids.shape[1] + 32)[0]["generated_text"]

    output = output[len(context):] # Remove the context string from the output
    output = output.replace(prompts["prefixes"][args.target_style], "") # Remove prefix
    output = output.split("\n")[0].strip()    # Remove anything after the first newline

    return output


def preprocess(text):
    # Preprocess for passing to OPT
    text = text.replace("\n", " ")
    return re.sub(r"(?<=\S) +(?=['.,;:!?])", "", text.strip())


def run(k, r):
    with open("{}/".format(DATA_DIR) +  paths["test"][args.source_style]["text"], "r", encoding="utf-8") as f:
        src_text = f.readlines()
    with open("{}/".format(DATA_DIR) + paths["train"][args.target_style]["text"], "r", encoding="utf-8") as f:
        tgt_text = f.readlines()

    input_path  = "{}/".format(OUTPUT_DIR) + args.source_style + "_to_" + args.target_style +  "_input_k{}_r{}.txt".format(k, r)
    output_path = "{}/".format(OUTPUT_DIR) + args.source_style + "_to_" + args.target_style + "_output_k{}_r{}.txt".format(k, r)

    with open(input_path,  "w") as f:
        f.write("")
    with open(output_path, "w") as f:
        f.write("") 

    # Embed and align the data
    aligned_src, aligned_tgt = aligned_embed(args, alignment_strength=1)

    print("Extracting k-nearest neighbors...") # Indices
    src_neighbors = [get_neighbors(src_vector, aligned_tgt, k) \
        for src_vector in tqdm.tqdm(aligned_src)]
    candidates = src_neighbors

    comb = {}
    for i, v in enumerate(candidates):
        comb[i] = subset(v, r)

    print("Generating... Check output file for progress...")
    for l, v in comb.items():
        # Retrieve the text at the closest neighbors
        src_neighbor_text = [[tgt_text[i] for i in neighbor_indices] \
            for neighbor_indices in v]

        # Join every list in src_neighbor_text into a single string
        src_neighbor_text = [" ".join(neighbor_text) for neighbor_text \
            in src_neighbor_text]

        options = []; scores = []; fallback = []
        curr_input = src_text[l]
        for i, e in enumerate(src_neighbor_text):
            src_neighbor_text = [preprocess(text) for text in src_neighbor_text]

            options.append(summarize(args, src_neighbor_text[i]))
            
            acc, cos_sim, cola_acc = evaluate(args.target_style, [curr_input], [e], float(args.lambda_score))
            scores.append(acc * cos_sim * cola_acc)
            fallback.append((acc + cos_sim + cola_acc) / 3)

        if np.sum(scores) == 0:
            best_option = options[np.argmax(fallback)]
            # print("Best Score (using Mean) {}".format(np.max(fallback)))
        else:
            best_option = options[np.argmax(scores)]
            # print("Best Score (using J) {}".format(np.max(scores)))

        with open(input_path,  "a") as f:
            f.write(curr_input)
        with open(output_path, "a") as f:
            f.write(best_option.replace("{", " ").replace("}", " ").strip())
            f.write("\n")
        clean()


if __name__ == "__main__":   
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=4, help="Objects (nearest neighbors)")
    parser.add_argument("--r", type=int, default=3, help="Subset size to summarize")
    parser.add_argument("--input",  type=str, default="", help="Input file")
    parser.add_argument("--source_style", type=str, default="", help="Source style")
    parser.add_argument("--target_style", type=str, default="", help="Target style")
    parser.add_argument("--lambda_score", type=str, default="", help="Lambda score")
    args = parser.parse_args()

    run(args.k, args.r)