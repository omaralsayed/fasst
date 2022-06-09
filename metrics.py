import numpy as np

import classifier
import embed
from utils import load_txt_or_bpe
import sys

import os
import gc
import tqdm
import torch
import argparse

# from fairseq.models.roberta import RobertaModel
# from fairseq.data.data_utils import collate_tokens

import re
import json

sys.path.append("../")
DIR = "./data"

with open("{}/".format(DIR) + "paths.json") as f:
    paths = json.load(f)


def detokenize(x):
    return re.sub(r" ([,\.;:?!])", r"\1", x)


def get_similarity(input_vector, output_vector):
    """Calculates the similarity between two embeddings.
    """
    return np.dot(input_vector, output_vector) / \
        (np.linalg.norm(input_vector) * np.linalg.norm(output_vector))


def get_similarity_score(input_strings, output_strings, input_embeddings=[], output_embeddings=[]):
    if input_embeddings  == [] or type(input_embeddings[0])  == str:
        input_embeddings  = embed.embed_text(input_strings)
    if output_embeddings == [] or type(output_embeddings[0]) == str:
        output_embeddings = embed.embed_text(output_strings)
    
    score = 0
    for i in range(len(input_embeddings)):
        score += get_similarity(input_embeddings[i], output_embeddings[i])

    return score / len(input_embeddings)


def get_accuracy_score(preds, target_style, embedding=[], model='centroids', lambda_score=0.0):
    """Calculates whether the sentence is correctly classified.
    """
    training_embedding = "{}/".format(DIR) +  paths["train"][target_style]["embedding"]

    if target_style == "formal" or target_style == "informal":
        out_dict = {"formal": 0, "informal": 0}
    else:
        out_dict = {"yelp_0": 0, "yelp_1": 0}

    print("Loading Centroids...")
    mean_vector_dict = classifier.load_style_mean_embeddings(training_embedding)

    print("Embedding Text...")
    raw_text = preds
    if embedding == [] or type(embedding[0]) == str:
        text_embeddings = embed.embed_text(raw_text)
    else:
        text_embeddings = embedding

    if model == "centroids":
        for i in range(len(raw_text)):
            style = classifier.classify_using_centroids(text_embeddings[i], mean_vector_dict)
            if style in out_dict:
                out_dict[style] += 1
            else:
                out_dict[style] = 1

    elif model == "tfidf_optimized":
        print("Loading TFIDF Dictionaries...")
        style_tfidf_dict = classifier.load_style_tfidf_dict()
        for i in range(len(raw_text)):
            style = classifier.classify_tfidf(text_embeddings[i], raw_text[i], mean_vector_dict, style_tfidf_dict, lambda_score)
            if style in out_dict:
                out_dict[style] += 1
            else:
                out_dict[style] = 1

    print(out_dict.keys())
    print("Target Style: {} Accuracy: {}".format(target_style, (out_dict[target_style] / len(raw_text))))
    return (out_dict[target_style] / len(raw_text))


# def get_cola_stats(preds, soft=False, batch_size=32):
#     print("Calculating Acceptability Score...")

#     path_to_data = "models/cola/cola-bin"

#     cola_classifier_path = "models/cola"
#     cola_checkpoint = "checkpoint_best.pt"

#     cola_roberta = RobertaModel.from_pretrained(
#         cola_classifier_path, 
#         checkpoint_file=cola_checkpoint,
#         data_name_or_path=path_to_data
#     )
#     cola_roberta.eval()

#     cola_stats = []
#     for i in tqdm.tqdm(range(0, len(preds), batch_size), total=len(preds) // batch_size):
#         sentences = preds[i:i + batch_size]

#         # Detokenize and BPE encode input
#         sentences = [cola_roberta.bpe.encode(detokenize(sent)) for sent in sentences]

#         batch = collate_tokens(
#             [cola_roberta.task.source_dictionary.encode_line("<s> " + sent + " </s>", 
#                 append_eos=False) for sent in sentences], pad_idx=1
#         )

#         batch = batch[:, :512]
#         with torch.no_grad():
#             predictions = cola_roberta.predict("sentence_classification_head", batch.long())

#         if soft:
#             prediction_labels = torch.softmax(predictions, axis=1)[:, 1].cpu().numpy()
#         else:
#             prediction_labels = predictions.argmax(axis=1).cpu().numpy()

#         cola_stats.extend(list(1 - prediction_labels))

#     return np.array(cola_stats)


def evaluate(target_style, inputs="", preds=""):
    print(target_style)
    print()
    print(type(target_style))
    print()
    if type(inputs[0]) != str:
        input_embeddings = inputs
    else:
        input_embeddings = inputs

    if type(preds[0])  != str:
        output_embeddings = preds
    else:
        output_embeddings = preds



    # cola_stats = get_cola_stats(preds)
    cola_score = 1.0 #sum(cola_stats) / len(preds)

    accuracy   = get_accuracy_score(preds, target_style, embedding=output_embeddings, model='tfidf_optimized', lambda_score=0.15)
    similarity = get_similarity_score(inputs, preds, input_embeddings, output_embeddings)
    
    print("ACC | SIM | FL |\n")
    print("--- | --- | -- |\n")
    print("{:.4f}|{:.4f}|{:.4f}|\n".format(accuracy, similarity, cola_score))



    # cola_stats = get_cola_stats(preds)
    # cola_score = sum(cola_stats) / len(preds)

    # accuracy   = get_accuracy_score(preds, target_style, embedding=output_embeddings, model='tfidf_optimized', lambda_score=0.10)
    # similarity = get_similarity_score(inputs, preds, input_embeddings, output_embeddings)
    
    # print("ACC | SIM | FL |\n")
    # print("--- | --- | -- |\n")
    # print("{:.4f}|{:.4f}|{:.4f}|\n".format(accuracy, similarity, cola_score))

    gc.collect()
    return accuracy, similarity, cola_score


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--inputs", help="Input path", required=True)
    parser.add_argument("-p", "--preds", help="Predictions path", required=True)
    parser.add_argument("-s", "--style", help="Target style", required=True)
    args = parser.parse_args()

    with open(args.inputs, "r") as input_file, open(args.preds, "r") as preds_file:
        inputs = input_file.readlines()
        preds  = preds_file.readlines()

    evaluate(args.style, inputs, preds)
