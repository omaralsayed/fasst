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

################# YELP ######################

### loading input
# directory to input
DIR_input_neg = "data/yelp_0/test.txt"
DIR_input_pos = "data/yelp_1/test.txt"

# THIS DIRECTORY NEED TO BE CHANGED AFTER THE DATA FOLDER I UPDATED
DIR_input_formal   = "../GYAFC_test/formal.txt"
DIR_input_informal = "../GYAFC_test/informal.txt"

with open(DIR_input_neg, "r") as input_neg_file, open(DIR_input_pos, "r") as input_pos_file:
    input_neg = input_neg_file.readlines()
    input_pos = input_pos_file.readlines()

with open(DIR_input_informal, "r") as input_informal_file, open(DIR_input_formal, "r") as input_formal_file:
    input_informal = input_informal_file.readlines()
    input_formal = input_formal_file.readlines()


# path to output data
DIR_YELP = "../OG_output/YELP"
DIR_GYAFC = "../OG_output/GYAFC"



def metrics_YELP(lambda_=0.15):

    # list of models with single output
    model_sgl = ["deep_latent_seq_He", "DualRL_Luo", "Styins_Yi"]

    # list of models with multiple output
    model_mlt = ["NAST_Huang", "StyTrans_Dai"]
    NAST = ["latentseq_learnable", "latentseq_simple", "stytrans_learnable", "stytrans_simple"]
    StyTrans = ["condition", "multi"]


    ######### MODELS with single output #############
    for model in model_sgl:
        # load model data
        # 0 -> 1 (neg -> pos)
        DIR_neg2pos = "{}/{}/".format(DIR_YELP, model) + "output_pos.txt"
        DIR_pos2neg = "{}/{}/".format(DIR_YELP, model) + "output_neg.txt"
        with open(DIR_neg2pos, "r") as neg2pos_file, open(DIR_pos2neg, "r") as pos2neg_file:
            output_neg2pos = neg2pos_file.readlines()
            output_pos2neg = pos2neg_file.readlines()
    
    
        ret_val1 = evaluate("yelp_0", input_pos, output_pos2neg, lambda_)
        ret_val2 = evaluate("yelp_1", input_neg, output_neg2pos, lambda_)
    
        # calculate the mean from both direction
        ret_mean = []
        for i,_ in enumerate(ret_val1):
            ret_mean.append((ret_val1[i]+ ret_val2[i])/2)
    
        print("------------------- MEAN OF BOTH DIRECTION-------------------")
        print("{}".format(model))
    
        print('| ACC | SIM | COS | BLEU | FL |  J  | mean | g2 | h2 |\n')
        print('| --- | --- | ... | ---- | -- | --- | ---- | -- | -- |\n')
    
        print('|{:.3f}|{:.3f}|{:.3f}|{:.3f}|{:.3f}|{:.3f}|{:.3f}|{:.3f}|{:.3f}|\n'.format(ret_mean[0], ret_mean[1], ret_mean[2], ret_mean[3], ret_mean[4],
            ret_mean[5], ret_mean[6], ret_mean[7], ret_mean[8] )
                )
    ######### MODELS with multiple outputs ###########
    for model in model_mlt:
        print("*"*50)
        print("Mlt MODELs")
        print("*"*50)
        # load model data
        # 0 -> 1 (neg -> pos)
        if model == "NAST_Huang": 
            model_list = NAST
        else:
            model_list = StyTrans
        for sub_model in model_list:
            DIR_neg2pos = "{}/{}/{}/".format(DIR_YELP, model, sub_model) + "output_pos.txt"
            DIR_pos2neg = "{}/{}/{}/".format(DIR_YELP, model, sub_model) + "output_neg.txt"
            with open(DIR_neg2pos, "r") as neg2pos_file, open(DIR_pos2neg, "r") as pos2neg_file:
                output_neg2pos = neg2pos_file.readlines()
                output_pos2neg = pos2neg_file.readlines()
    
    
            ret_val1 = evaluate("yelp_0", input_pos, output_pos2neg, lambda_)
            ret_val2 = evaluate("yelp_1", input_neg, output_neg2pos, lambda_)
    
            # calculate the mean from both direction
            ret_mean = []
            for i,_ in enumerate(ret_val1):
                ret_mean.append((ret_val1[i]+ ret_val2[i])/2)
    
            print("------------------- MEAN OF BOTH DIRECTION-------------------")
            print("{}; {}".format(model, sub_model))
    
            print('| ACC | SIM | COS | BLEU | FL |  J  | mean | g2 | h2 |\n')
            print('| --- | --- | ... | ---- | -- | --- | ---- | -- | -- |\n')
    
            print('|{:.3f}|{:.3f}|{:.3f}|{:.3f}|{:.3f}|{:.3f}|{:.3f}|{:.3f}|{:.3f}|\n'.format(ret_mean[0], ret_mean[1], ret_mean[2], ret_mean[3], ret_mean[4],
                ret_mean[5], ret_mean[6], ret_mean[7], ret_mean[8] )
                    )


def metrics_GYAFC(lambda_=0.029):

    # list of models with single output
    model_sgl = ["DualRL_Luo"]

    # list of models with multiple output
    model_mlt = ["NAST_Huang", "StyIns_Yi"]
    NAST = ["latentseq_learnable", "latentseq_simple", "stytrans_learnable", "stytrans_simple"]
    StyIns = ["2.5k", "10k", "unsuper"]

    
    ######### MODELS with single output #############
    for model in model_sgl:
        # load model data
        # 0 -> 1 (neg -> pos)
        DIR_informal2formal = "{}/{}/".format(DIR_GYAFC, model) + "output_formal.txt"
        DIR_formal2informal = "{}/{}/".format(DIR_GYAFC, model) + "output_informal.txt"
        with open(DIR_informal2formal, "r") as informal2formal_file, open(DIR_formal2informal, "r") as formal2informal_file:
            output_informal2formal = informal2formal_file.readlines()
            output_formal2informal = formal2informal_file.readlines()

        if model == "DualRL_Luo":
            DIR_formal = "{}/{}/".format(DIR_GYAFC, model) + "input_formal.txt"
            DIR_informal = "{}/{}/".format(DIR_GYAFC, model) + "input_informal.txt"
            with open(DIR_formal, "r") as formal_file, open(DIR_informal, "r") as informal_file:
                input_formal_DualRL = formal_file.readlines()
                input_informal_DualRL = informal_file.readlines()
            

            ret_val1 = evaluate("informal", input_formal_DualRL, output_formal2informal, lambda_)
            ret_val2 = evaluate("formal", input_informal_DualRL, output_informal2formal, lambda_)

        else:
            ret_val1 = evaluate("informal", input_formal, output_formal2informal, lambda_)
            ret_val2 = evaluate("formal", input_informal, output_informal2formal, lambda_)
        
        # calculate the mean from both direction
        ret_mean = []
        for i,_ in enumerate(ret_val1):
            ret_mean.append((ret_val1[i]+ ret_val2[i])/2)

        print("------------------- MEAN OF BOTH DIRECTION-------------------")
        print("{}".format(model))

        print('| ACC | SIM | COS | BLEU | FL |  J  | mean | g2 | h2 |\n')
        print('| --- | --- | ... | ---- | -- | --- | ---- | -- | -- |\n')

        print('|{:.3f}|{:.3f}|{:.3f}|{:.3f}|{:.3f}|{:.3f}|{:.3f}|{:.3f}|{:.3f}|\n'.format(ret_mean[0], ret_mean[1], ret_mean[2], ret_mean[3], ret_mean[4],
            ret_mean[5], ret_mean[6], ret_mean[7], ret_mean[8] )
                )
    ######### MODELS with multiple outputs ###########
    for model in model_mlt:
        print("*"*50)
        print("Mlt MODELs")
        print("*"*50)
        # load model data
        # 0 -> 1 (neg -> pos)
        if model == "NAST_Huang":
            model_list = NAST
        else:
            model_list = StyIns
        for sub_model in model_list:
            DIR_informal2formal = "{}/{}/{}/".format(DIR_GYAFC, model, sub_model) + "output_formal.txt"
            DIR_formal2informal = "{}/{}/{}/".format(DIR_GYAFC, model, sub_model) + "output_informal.txt"
            with open(DIR_informal2formal, "r") as informal2formal_file, open(DIR_formal2informal, "r") as formal2informal_file:
                output_informal2formal = informal2formal_file.readlines()
                output_formal2informal = formal2informal_file.readlines()
                
            print("Check size")
            print(len(input_formal))
            print(len(output_formal2informal))


            ret_val1 = evaluate("informal", input_formal, output_formal2informal, lambda_)
            ret_val2 = evaluate("formal", input_informal, output_informal2formal, lambda_)

            # calculate the mean from both direction
            ret_mean = []
            for i,_ in enumerate(ret_val1):
                ret_mean.append((ret_val1[i]+ ret_val2[i])/2)

            print("------------------- MEAN OF BOTH DIRECTION-------------------")
            print("{}; {}".format(model, sub_model))

            print('| ACC | SIM | COS | BLEU | FL |  J  | mean | g2 | h2 |\n')
            print('| --- | --- | ... | ---- | -- | --- | ---- | -- | -- |\n')

            print('|{:.3f}|{:.3f}|{:.3f}|{:.3f}|{:.3f}|{:.3f}|{:.3f}|{:.3f}|{:.3f}|\n'.format(ret_mean[0], ret_mean[1], ret_mean[2], ret_mean[3], ret_mean[4],
                ret_mean[5], ret_mean[6], ret_mean[7], ret_mean[8] )
                    )




if __name__=="__main__":
    
    metrics_GYAFC(lambda_=0.029)

    #metrics_YELP(lambda_=0.15)

    sys.exit()



