import os
import gc
import sys
import torch
from metrics_mlt_scores import evaluate

def clean():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

clean()


#combinations:
# 33, 44, 43, 53, 54
combination = [(3,3), (4,4), (4,3), (5,3), (5,4)]

def metrics_yelp(lambda_ = 0.166):

    in_style =0
    out_style =1

    for comb in combination:
        in_style =0
        out_style =1
        
        k = comb[0]
        r = comb[1]
        
        
        inputs  = f"fasst_final_outputs/fasst_yelp/yelp_{in_style}_to_yelp_{out_style}_input_k{k}_r{r}.txt"
        outputs = f"fasst_final_outputs/fasst_yelp/yelp_{in_style}_to_yelp_{out_style}_output_k{k}_r{r}.txt"
        target_style = "yelp_{}".format(out_style)
    
        with open(inputs, "r") as input_file, open(outputs, "r") as outputs_file:
            inputs = input_file.readlines()
            outputs  = outputs_file.readlines()
        

        #print("k={},r={};{} to {} len".format(k,r,in_style,out_style), len(outputs))
        print("*"*50)
        print("style {} to style {} for k={}, r={}, len={}".format(in_style, out_style, k,r, len(outputs)))
        print("*"*50)
    
        ret_val1 = evaluate(target_style, inputs, outputs, lambda_)
        
        
        inputs  = f"output/yelp_{out_style}_to_yelp_{in_style}_input_k{k}_r{r}.txt"
        outputs = f"output/yelp_{out_style}_to_yelp_{in_style}_output_k{k}_r{r}.txt"
        target_style = "yelp_{}".format(in_style)
        
        with open(inputs, "r") as input_file, open(outputs, "r") as outputs_file:
            inputs = input_file.readlines()
            outputs  = outputs_file.readlines()
        
        print("*"*50)
        print("style {} to style {} for k={}, r={}, len={}".format(out_style, in_style, k,r, len(outputs)))
        print("*"*50)
    
        ret_val2 = evaluate(target_style, inputs, outputs, lambda_)
        
        # calculate the mean from both direction
        ret_mean = []
        for i,_ in enumerate(ret_val1):
            ret_mean.append((ret_val1[i]+ ret_val2[i])/2)
        
        print("------------------- MEAN OF BOTH DIRECTION-------------------")
        print("for k={}, r={}".format(k,r))
        print("file length", len(outputs))
        
        print('| ACC | SIM | COS | BLEU | FL |  J  | mean | g2 | h2 |\n')
        print('| --- | --- | ... | ---- | -- | --- | ---- | -- | -- |\n')
    
        print('|{:.3f}|{:.3f}|{:.3f}|{:.3f}|{:.3f}|{:.3f}|{:.3f}|{:.3f}|{:.3f}|\n'.format(ret_mean[0], ret_mean[1], ret_mean[2], ret_mean[3], ret_mean[4],
            ret_mean[5], ret_mean[6], ret_mean[7], ret_mean[8] )
                )

def metrics_GYAFC(lambda_ = 0.115):

    style_formal = "formal"
    style_informal = "informal"

    for comb in combination:
        in_style =0
        out_style =1

        k = comb[0]
        r = comb[1]


        inputs  = f"fasst_final_outputs/fasst_form/formal_to_informal_input_k{k}_r{r}.txt"
        outputs = f"fasst_final_outputs/fasst_form/formal_to_informal_output_k{k}_r{r}.txt"
        #target_style = "{}".format(style_informal)
        target_style = "informal"


        with open(inputs, "r") as input_file, open(outputs, "r", encoding="ISO-8859-1") as outputs_file:
            inputs = input_file.readlines()
            outputs  = outputs_file.readlines()


        #print("k={},r={};{} to {} len".format(k,r,in_style,out_style), len(outputs))
        print("*"*50)
        print("formal to informal for k={}, r={}, len={}".format(k,r, len(outputs)))
        print("*"*50)

        ret_val1 = evaluate(target_style, inputs, outputs, lambda_)


        #inputs  = f"output/yelp_{out_style}_to_yelp_{in_style}_input_k{k}_r{r}.txt"
        #outputs = f"output/yelp_{out_style}_to_yelp_{in_style}_output_k{k}_r{r}.txt"
        #target_style = "yelp_{}".format(in_style)
        inputs  = f"fasst_final_outputs/fasst_form/informal_to_formal_input_k{k}_r{r}.txt"
        outputs = f"fasst_final_outputs/fasst_form/informal_to_formal_output_k{k}_r{r}.txt"
        #target_style = "{}".format(style_formal)
        target_style = "formal"




        with open(inputs, "r") as input_file, open(outputs, "r") as outputs_file:
            inputs = input_file.readlines()
            outputs  = outputs_file.readlines()

        print("*"*50)
        print("informal to formal for k={}, r={}, len={}".format(k,r, len(outputs)))
        print("*"*50)

        ret_val2 = evaluate(target_style, inputs, outputs, lambda_)

        # calculate the mean from both direction
        ret_mean = []
        for i,_ in enumerate(ret_val1):
            ret_mean.append((ret_val1[i]+ ret_val2[i])/2)

        print("------------------- MEAN OF BOTH DIRECTION-------------------")
        print("for k={}, r={}".format(k,r))
        print("file length", len(outputs))

        print('| ACC | SIM | COS | BLEU | FL |  J  | mean | g2 | h2 |\n')
        print('| --- | --- | ... | ---- | -- | --- | ---- | -- | -- |\n')

        print('|{:.3f}|{:.3f}|{:.3f}|{:.3f}|{:.3f}|{:.3f}|{:.3f}|{:.3f}|{:.3f}|\n'.format(ret_mean[0], ret_mean[1], ret_mean[2], ret_mean[3], ret_mean[4],
            ret_mean[5], ret_mean[6], ret_mean[7], ret_mean[8] )
                )


if __name__=="__main__":


   #metrics_yelp() 
   metrics_GYAFC() 

sys.exit()

