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
DIR_input_neg = "../Luo_output/YELP/"
DIR_input_pos = "../Luo_output/YELP/"

DIR_input_formal   = "../Luo_output/GYAFC/"
DIR_input_informal = "../Luo_output/GYAFC/"



# path to output data
DIR_YELP = "../Luo_output/YELP"
DIR_GYAFC = "../Luo_output/GYAFC"





def run_metrics(lambda_=0.15, data = "yelp"):

    # list of models 
    model_list = ["BackTranslation_Pr", "CrossAlignment_Shen", "DeleteOnly_Li", "DeleteRetrieve_Li", "Multidecoder_Fu", "RetrieveOnly_Li", "StyleEmbedding_Fu", "TemplateBase_Li", "UnpairedRL_Xu", "UnsuperMT_Zhang"]

    
    for model in model_list:
        with open(DIR_input_neg + model+ "/input_neg.txt", "r") as input_neg_file, open(DIR_input_pos + model + "/input_neg.txt", "r") as input_pos_file:
            input_neg = input_neg_file.readlines()
            input_pos = input_pos_file.readlines()

        with open(DIR_input_informal + model + "/input_informal.txt", "r") as input_informal_file, open(DIR_input_formal + model + "/input_formal.txt", "r") as input_formal_file:
            input_informal = input_informal_file.readlines()
            input_formal = input_formal_file.readlines()


        # load model data
        if data=="yelp":
            print("YELP")
            DIR_neg2pos = "{}/{}/".format(DIR_YELP, model) + "output_pos.txt"
            DIR_pos2neg = "{}/{}/".format(DIR_YELP, model) + "output_neg.txt"
            with open(DIR_neg2pos, "r") as neg2pos_file, open(DIR_pos2neg, "r") as pos2neg_file:
                output_neg2pos = neg2pos_file.readlines()
                output_pos2neg = pos2neg_file.readlines()

            ret_val1 = evaluate("yelp_0", input_pos, output_pos2neg, lambda_)
            ret_val2 = evaluate("yelp_1", input_neg, output_neg2pos, lambda_)
        
        else:
            print("GYAFC")
            DIR_formal = "{}/{}/".format(DIR_GYAFC, model) + "input_formal.txt"
            DIR_informal = "{}/{}/".format(DIR_GYAFC, model) + "input_informal.txt"
            with open(DIR_formal, "r") as formal_file, open(DIR_informal, "r") as informal_file:
                input_formal = formal_file.readlines()
                input_informal = informal_file.readlines()

            ret_val1 = evaluate("informal", input_formal_DualRL, output_formal2informal, lambda_)
            ret_val2 = evaluate("formal", input_informal_DualRL, output_informal2formal, lambda_)



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







if __name__=="__main__":
    

    run_metrics(lambda_=0.15)

    sys.exit()



