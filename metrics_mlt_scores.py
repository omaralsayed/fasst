import numpy as np

import classifier
import embed
import sys

import os
import gc
import tqdm
import torch
import argparse

from nltk.translate.bleu_score import sentence_bleu

from fairseq.models.roberta import RobertaModel
from fairseq.data.data_utils import collate_tokens

import re
import json
import warnings
warnings.filterwarnings("ignore")


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


def add_spacing(s):
    s = re.sub('([.,!?()])', r' \1 ', s)
    return re.sub('\s{2,}', ' ', s)


# Disable
def blockPrint():
    sys.stdout = open(os.devnull, 'w')

# Restore
def enablePrint():
    sys.stdout = sys.__stdout__


def get_similarity_score(input_strings, output_strings, input_embeddings=[], output_embeddings=[]):
    if input_embeddings  == [] or type(input_embeddings[0])  == str:
        input_embeddings  = embed.embed_text(input_strings)
    if output_embeddings == [] or type(output_embeddings[0]) == str:
        output_embeddings = embed.embed_text(output_strings)
    
    score_vector = np.array([])
    score = 0
    for i in range(len(input_embeddings)):
        scr = get_similarity(input_embeddings[i], output_embeddings[i])
        score_vector = np.append(score_vector,scr)
        score += scr

    return score / len(input_embeddings), score_vector


def get_accuracy_score(preds, target_style, embedding=[], model='tf_idf_optimized' #'centroids'
        , lambda_score=0.0):
    """Calculates whether the sentence is correctly classified.
    """   
    # print("Loading Centroids...")
#    print("Calculate acc...")
    mean_vector_dict = classifier.load_style_mean_embeddings(target_style)

    # print("Embedding Text...")
    raw_text = preds
    if embedding == [] or type(embedding[0]) == str:
        text_embeddings = embed.embed_text(raw_text)
    else:
        text_embeddings = embedding

    if target_style == "formal" or target_style == "informal":
        out_dict = {"formal": 0, "informal": 0}
        #out_dict_vec = {"formal": np.array([]) , "informal": 0}

    else:
        out_dict = {"yelp_0": 0, "yelp_1": 0}

    #acc_vector_dict =dict() # np.array([])
    acc_vector = np.array([])
    if model == "centroids":
        for i in range(len(raw_text)):
            style = classifier.classify_using_centroids(text_embeddings[i], mean_vector_dict)
            if style == target_style:
                acc_vector = np.append(acc_vector,1)
            else:
                acc_vector = np.append(acc_vector,0)
            
            if style in out_dict:
                out_dict[style] += 1
                #acc_vector = np.append(acc_vector,1)
            else:
                out_dict[style] = 1
                #acc_vector = np.append(acc_vector,0)

                print(acc_vector)


    elif model == "tfidf_optimized":
        # print("Loading TFIDF Dictionaries...")
        style_tfidf_dict = classifier.load_style_tfidf_dicts(target_style)
        for i in range(len(raw_text)):
            style = classifier.classify_tfidf(text_embeddings[i], raw_text[i], mean_vector_dict, style_tfidf_dict, lambda_score)
            if style == target_style:
                acc_vector = np.append(acc_vector,1)
            else:
                acc_vector = np.append(acc_vector,0)
            
            if style in out_dict:
                out_dict[style] += 1

            else:
                out_dict[style] = 1

    return (out_dict[target_style] / len(raw_text)), acc_vector

def get_cola_stats(preds, soft=False, batch_size=32):
    """ 
    based on:
    https://github.com/martiansideofthemoon/style-transfer-paraphrase/blob/master/style_paraphrase/evaluation/scripts/roberta_classify.py
    https://github.com/skoltech-nlp/detox
    """
    path_to_data = "models/cola/cola-bin"

    cola_classifier_path = "models/cola"
    cola_checkpoint = "checkpoint_best.pt"

    cola_roberta = RobertaModel.from_pretrained(
        cola_classifier_path, 
        checkpoint_file=cola_checkpoint,
        data_name_or_path=path_to_data
    )
    cola_roberta.eval()

    cola_stats = []
    for i in tqdm.tqdm(range(0, len(preds), batch_size), total=len(preds) // batch_size):
        sentences = preds[i:i + batch_size]

        # Detokenize and BPE encode input
        sentences = [cola_roberta.bpe.encode(detokenize(sent)) for sent in sentences]

        batch = collate_tokens(
            [cola_roberta.task.source_dictionary.encode_line("<s> " + sent + " </s>", 
                append_eos=False) for sent in sentences], pad_idx=1
        )

        batch = batch[:, :512]
        with torch.no_grad():
            predictions = cola_roberta.predict("sentence_classification_head", batch.long())

        if soft:
            prediction_labels = torch.softmax(predictions, axis=1)[:, 1].cpu().numpy()
        else:
            prediction_labels = predictions.argmax(axis=1).cpu().numpy()

        cola_stats.extend(list(1 - prediction_labels))

    return np.array(cola_stats)


def evaluate(target_style, inputs="", preds="", lambda_score=0.15):
    if type(inputs[0]) != str:
        input_embeddings = inputs
    else:
        input_embeddings = inputs

    if type(preds[0])  != str:
        output_embeddings = preds
    else:
        output_embeddings = preds

    cola_stats = get_cola_stats(preds)
    cola_score = sum(cola_stats) / len(preds)

    accuracy, acc_vector   = get_accuracy_score(preds, target_style, embedding=output_embeddings, model='tfidf_optimized', lambda_score=lambda_score)
    cos_similarity, cos_similarity_vector= get_similarity_score(inputs, preds, input_embeddings, output_embeddings)
    
    #### J, mean, gmean anf hmean

    # joint
    J3 = accuracy*cos_similarity*cola_score
    J2 = accuracy*cos_similarity



    # mean
    mean3 = (accuracy+ cos_similarity+cola_score)/3
    mean2 = (accuracy+ cos_similarity)/2

    # G2
    gmean =np.sqrt(accuracy*cos_similarity*cola_score)

    # H2
    hmean = 3/(1/accuracy + 1/cos_similarity + 1/cola_score)


##    # write res to table
##    print('| ACC | SIM | COS | FL |  J3 | J2 |  mean3| mean2| g2 | h2 |\n')
##    print('| --- | --- | ... | -- | --- | -- |  ---- | ---- | -- | -- |\n')
##    print('|{:.3f}|{:.3f}|{:.3f}|{:.3f}|{:.3f}|{:.3f}|{:.3f}|{:.3f}|{:.3f}|{:.3f}|\n'.format(accuracy, avg_sim_by_sent, cos_similarity, cola_score, J3, J2, mean3, mean2,gmean,hmean))




    gc.collect()
    return accuracy, cos_similarity, cola_score, J3, J2, mean3, mean2, gmean,hmean


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--inputs", help="Input path", required=True)
    parser.add_argument("-p", "--preds", help="Predictions path", required=True)
    parser.add_argument("-s", "--style", help="Target style", required=True)
    args = parser.parse_args()

    with open(args.inputs, "r") as input_file, open(args.preds, "r") as preds_file:   
        inputs_ = input_file.readlines()
        preds_  = preds_file.readlines()
        print("og, input length", len(inputs_))
        print("og, output length", len(preds_))
        inputs = []
        preds  = []
        for i, line in enumerate(preds_):
            if len(line)>2:
                inputs.append(inputs_[i])
                preds.append(preds_[i])
        
        print("dlt, input length", len(inputs))
        print("dlt, output length", len(preds))



    evaluate(args.style, inputs, preds)
