import os
import gc
import sys
import torch
from metrics_mlt_scores import evaluate

def clean():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


#combinations:
# 33, 44, 43, 53, 54


clean()


in_style =0
out_style =1
combination = [(3,3), (4,4), (4,3), (5,3), (5,4)]
for comb in combination:
    k = comb[0]
    r = comb[1]
    lambda_ = 0.15 
    
    inputs  = f"output/yelp_{in_style}_to_yelp_{out_style}_input_k{k}_r{r}.txt"
    outputs = f"output/yelp_{in_style}_to_yelp_{out_style}_output_k{k}_r{r}.txt"
    target_style = "yelp_{}".format(out_style)

    with open(inputs, "r") as input_file, open(outputs, "r") as outputs_file:
        inputs = input_file.readlines()
        outputs  = outputs_file.readlines()
    
    print("*"*50)
    print("style {} to style {} for k={}, r={}".format(in_style, out_style, k,r))
    print("*"*50)

    ret_val1 = evaluate(target_style, inputs, outputs, lambda_)
    
    
    inputs  = f"output/yelp_{out_style}_to_yelp_{in_style}_input_k{k}_r{r}.txt"
    outputs = f"output/yelp_{out_style}_to_yelp_{in_style}_output_k{k}_r{r}.txt"
    target_style = "yelp_{}".format(in_style)
    
    with open(inputs, "r") as input_file, open(outputs, "r") as outputs_file:
        inputs = input_file.readlines()
        outputs  = outputs_file.readlines()
    
    print("*"*50)
    print("style {} to style {} for k={}, r={}".format(out_style, in_style, k,r))
    print("*"*50)

    ret_val2 = evaluate(target_style, inputs, outputs, lambda_)
    
    # calculate the mean from both direction
    ret_mean = []
    for i,_ in enumerate(ret_val1):
        ret_mean.append((ret_val1[i]+ ret_val2[i])/2)
    
    print("------------------- MEAN OF BOTH DIRECTION-------------------")
    print("for k={}, r={}".format(k,r))
    
    print('| ACC | SIM | COS | BLEU | FL |  J  | mean | g2 | h2 |\n')
    print('| --- | --- | ... | ---- | -- | --- | ---- | -- | -- |\n')

    print('|{:.3f}|{:.3f}|{:.3f}|{:.3f}|{:.3f}|{:.3f}|{:.3f}|{:.3f}|{:.3f}|\n'.format(ret_mean[0], ret_mean[1], ret_mean[2], ret_mean[3], ret_mean[4],
        ret_mean[5], ret_mean[6], ret_mean[7], ret_mean[8] )
            )



sys.exit()

