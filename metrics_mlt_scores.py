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
from wieting_similarity.similarity_evaluator import SimilarityEvaluator

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

            ##if style in out_dict:
            ##    acc_vector_dict[style] = np.append(acc_vector_dict[style],1)
            ##    out_dict[style] += 1
            ##else:
            ##    acc_vector_dict[style] = np.array([1])
            ##    out_dict[style] = 1


    # print("Target Style: {} Accuracy: {}".format(target_style, (out_dict[target_style] / len(raw_text))))
    return (out_dict[target_style] / len(raw_text)), acc_vector

def calc_bleu(inputs, preds):
    bleu_sim = 0
    counter = 0
 #   print('Calculating BLEU similarity')
    for i in range(len(inputs)):
        if len(inputs[i]) > 3 and len(preds[i]) > 3:
            bleu_sim += sentence_bleu([inputs[i]], preds[i])
            counter += 1

    return float(bleu_sim / counter)


def get_cola_stats(preds, soft=False, batch_size=32):
  #  print("Calculating Acceptability Score...")

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

def wieting_sim(inputs, preds, batch_size= 32):
    assert len(inputs) == len(preds)
    #print(len(inputs))
    #print(len(preds))

   # print('Calculating similarity by Wieting subword-embedding SIM model')
    #sys.exit()

    sim_evaluator = SimilarityEvaluator()

    sim_scores = []

    for i in tqdm.tqdm(range(0, len(inputs), batch_size)):
        sim_scores.extend(
            sim_evaluator.find_similarity(inputs[i:i + batch_size], preds[i:i + batch_size])
        )

    return np.array(sim_scores)



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
    
    #print("acc",accuracy)
    #print("acc_vec",sum(acc_vector)/len(acc_vector))

    
    #print("cos_sim_scr",cos_similarity)
    #print("cos_sim_vec",sum(cos_similarity_vector)/len(cos_similarity_vector))

    ## bleu score
    
    # Disable
    blockPrint()
    bleu = calc_bleu(inputs, preds)
    # Restore
    enablePrint()
    
    ## SIM by sent
    similarity_by_sent = wieting_sim(inputs, preds)
    avg_sim_by_sent = similarity_by_sent.mean()



    #### J, mean, g2 anf h2

    # joint
    J = sum(acc_vector*cos_similarity_vector*cola_stats)/len(preds)
    # mean
    mean = sum((acc_vector + cos_similarity_vector+cola_stats)/3)/len(preds)
    # G2
    gmean =sum(np.sqrt(abs(acc_vector*cos_similarity_vector*cola_stats)))/len(preds)
    # H2
    hmean = sum(3/(1/acc_vector + 1/cos_similarity_vector + 1/cola_stats))/len(preds)

###    # write res to table
###    print('| ACC | SIM | COS | BLEU | FL |  J  | mean | g2 | h2 |\n')
###    print('| --- | --- | ... | ---- | -- | --- | ---- | -- | -- |\n')
###    print('|{:.3f}|{:.3f}|{:.3f}|{:.3f}|{:.3f}|{:.3f}|{:.3f}|{:.3f}|{:.3f}|\n'.format(accuracy, avg_sim_by_sent, cos_similarity, bleu, cola_score, J, mean,gmean,hmean))




    gc.collect()
    return accuracy, avg_sim_by_sent, cos_similarity, bleu, cola_score, J, mean,gmean,hmean


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
