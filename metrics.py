import argparse
import json
import os
import re
import sys
import warnings

import numpy as np
import torch
import tqdm
from fairseq.data.data_utils import collate_tokens
from fairseq.models.roberta import RobertaModel

import classifier
import embed
from utils import get_similarity

warnings.filterwarnings("ignore")


sys.path.append("../")
with open("./data/paths.json") as f:
    paths = json.load(f)

def detokenize(s):
    """
    Detokenize a string by removing spaces before punctuation.

    Args:
        s (str): The tokenized string.

    Returns:
        str: The detokenized string.
    """
    return re.sub(r" ([,\.;:?!])", r"\1", s)

def add_spacing(s):
    """
    Add spacing around punctuation marks and reduce spaces.

    Args:
        s (string): The input string.
    """
    s = re.sub('([.,!?()])', r' \1 ', s)
    return re.sub('\s{2,}', ' ', s).strip()

def blockPrint():
    """
    Block printing to stdout by redirecting output to devnull.
    """
    sys.stdout = open(os.devnull, 'w')

def enablePrint():
    """
    Enable printing to stdout by restoring the default stdout.
    """
    sys.stdout = sys.__stdout__

def get_similarity_score(input_strings, output_strings, input_embeddings=None, output_embeddings=None):
    """
    Calculate the average similarity score and individual similarity scores 
    between two sets of strings or their embeddings.

    Args:
        input_strings (list of str): List of original input strings.
        output_strings (list of str): List of output strings to compare against the input.
        input_embeddings (list, optional): The embeddings of the input strings. Defaults to None.
        output_embeddings (list, optional): The embeddings of the output strings. Defaults to None.

    Returns:
        tuple: A tuple containing the average similarity score (float) and an array of individual similarity scores (numpy.ndarray).
    """
    if input_embeddings is None or isinstance(input_embeddings[0], str):
        input_embeddings = embed.embed_text(input_strings)
    if output_embeddings is None or isinstance(output_embeddings[0], str):
        output_embeddings = embed.embed_text(output_strings)
    
    score_vector = np.array([get_similarity(input_emb, output_emb) for input_emb, output_emb in zip(input_embeddings, output_embeddings)])
    average_score = np.mean(score_vector) if len(input_embeddings) > 0 else 0

    return average_score, score_vector

def get_accuracy_score(preds, target_style, embedding=None, model='tf_idf_optimized', lambda_score=0.0):
    """
    Calculate the accuracy score for predicted text against a target style using specified classification model.

    Args:
        preds (list of str): Predicted sentences or texts.
        target_style (str): The target style to compare against.
        embedding (list, optional): Precomputed embeddings for the predicted sentences. Defaults to None.
        model (str, optional): The model used for classification ('centroids' or 'tfidf_optimized'). Defaults to 'centroids'.
        lambda_score (float, optional): Lambda score for TF-IDF optimized classification. Defaults to None.

    Returns:
        tuple: A tuple containing the accuracy score (float) and an array of individual classification results (numpy.ndarray).
    """
    mean_vector_dict = classifier.load_style_mean_embeddings(target_style)

    text_embeddings = embed.embed_text(preds) if embedding is None or isinstance(embedding[0], str) else embedding

    relevant_styles = ["formal", "informal"] if target_style in ["formal", "informal"] else ["yelp_0", "yelp_1"]
    out_dict = {style: 0 for style in relevant_styles}

    acc_vector = []
    if model == "centroids":
        classify = lambda emb: classifier.classify_using_centroids(emb, mean_vector_dict)
    elif model == "tfidf_optimized":
        style_tfidf_dict = classifier.load_style_tfidf_dicts(target_style)
        classify = lambda emb, raw: classifier.classify_tfidf(emb, raw, mean_vector_dict, style_tfidf_dict, lambda_score)
    else:
        raise ValueError(f"Unsupported model: {model}")

    for i, emb in enumerate(text_embeddings):
        style = classify(emb, preds[i]) if model == "tfidf_optimized" else classify(emb)
        acc_vector.append(1 if style == target_style else 0)
        out_dict[style] = out_dict.get(style, 0) + 1

    accuracy = out_dict[target_style] / len(preds) if preds else 0
    return accuracy, np.array(acc_vector)

def get_cola_stats(preds, soft=False, batch_size=32):
    """
    Evaluate the "grammaticality" of predictions using the CoLA model.

    Ref:

        1. https://github.com/martiansideofthemoon/style-transfer-paraphrase/blob/master/style_paraphrase/evaluation/scripts/roberta_classify.py
        2. https://github.com/skoltech-nlp/detox
    
    Args:
        preds (list of str): Predicted sentences to be evaluated.
        soft (bool, optional): Whether to use soft labels based on softmax. Defaults to False.
        batch_size (int, optional): Batch size for processing. Defaults to 32.
    
    Returns:
        numpy.ndarray: Array of grammatical correctness scores for each prediction.
    """
    path_to_data = "models/cola/cola-bin"
    cola_classifier_path = "models/cola"
    cola_checkpoint = "checkpoint_best.pt"

    # Load the RoBERTa model for the CoLA task
    cola_roberta = RobertaModel.from_pretrained(
        cola_classifier_path, 
        checkpoint_file=cola_checkpoint,
        data_name_or_path=path_to_data
    )
    cola_roberta.eval()

    cola_stats = []
    for start_idx in tqdm.tqdm(range(0, len(preds), batch_size), total=(len(preds) + batch_size - 1) // batch_size):
        sentences = preds[start_idx:start_idx + batch_size]
        sentences = [cola_roberta.bpe.encode(detokenize(sent)) for sent in sentences]

        # Prepare the batch for classification
        batch = collate_tokens(
            [cola_roberta.task.source_dictionary.encode_line("<s> " + sent + " </s>", append_eos=False) for sent in sentences], pad_idx=1
        ).long()[:512]  # Limit to 512 tokens

        with torch.no_grad():
            predictions = cola_roberta.predict("sentence_classification_head", batch)

        prediction_labels = torch.softmax(predictions, axis=1)[:, 1].cpu().numpy() if soft else predictions.argmax(axis=1).cpu().numpy()
        cola_stats.extend(1 - prediction_labels)

    return np.array(cola_stats)

def evaluate(target_style, inputs="", preds="", lambda_score=0.15):
    """
    Evaluate the performance of predictions against a target style using various metrics.

    Args:
        target_style (str): The target style to evaluate against.
        inputs (list of str or embeddings): Original sentences or their embeddings.
        preds (list of str or embeddings): Predicted sentences or their embeddings.
        lambda_score (float): Lambda score for TF-IDF optimization.

    Returns:
        tuple: A tuple containing various evaluation metrics.
    """
    input_embeddings = inputs
    output_embeddings = preds

    cola_stats = get_cola_stats(preds)
    cola_score = np.mean(cola_stats) if preds else 0

    accuracy, _ = get_accuracy_score(preds, target_style, embedding=output_embeddings, model='tfidf_optimized', lambda_score=lambda_score)
    cos_similarity, _ = get_similarity_score(inputs, preds, input_embeddings, output_embeddings)

    J3 = accuracy * cos_similarity * cola_score
    J2 = accuracy * cos_similarity

    mean3 = np.mean([accuracy, cos_similarity, cola_score])
    mean2 = np.mean([accuracy, cos_similarity])

    gmean = np.sqrt(accuracy * cos_similarity * cola_score) if all([accuracy, cos_similarity, cola_score]) else 0
    hmean = 3 / (sum(1/x for x in [accuracy, cos_similarity, cola_score] if x)) if any([accuracy, cos_similarity, cola_score]) else 0

    return accuracy, cos_similarity, cola_score, J3, J2, mean3, mean2, gmean, hmean

def main():
    parser = argparse.ArgumentParser(description="Evaluate predictions against a target style.")
    parser.add_argument("-i", "--inputs", help="Path to the input file containing original sentences.", required=True)
    parser.add_argument("-p", "--preds", help="Path to the predictions file.", required=True)
    parser.add_argument("-s", "--style", help="Target style for evaluation.", required=True)
    args = parser.parse_args()

    with open(args.inputs, "r", encoding="utf-8") as input_file, open(args.preds, "r", encoding="utf-8") as preds_file:
        inputs_, preds_ = input_file.readlines(), preds_file.readlines()
        
        # Filter out pairs where the prediction line length is <= 2
        inputs, preds = zip(*[(input_line, pred_line) for input_line, pred_line in zip(inputs_, preds_) if len(pred_line) > 2])

    # Evaluate using the filtered inputs and predictions
    evaluate(args.style, list(inputs), list(preds))

if __name__ == "__main__":
    main()
