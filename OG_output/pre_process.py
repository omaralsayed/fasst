import numpy as np
import os
import sys
import csv
import re

# create new directory
directory ="seperated_in_out"
if not os.path.isdir(directory):
    os.mkdir(directory)

# formal to informal
with open("neg2pos.txt", "r") as f:
    test01= f.readlines()


with open(directory + "/input_neg.txt","w") as f:
    for i, line in enumerate(test01):
        splitted_lines = line.split('\t')
        f.write(splitted_lines[0].strip()+ "\n")   
    
with open(directory +"/output_pos.txt","w") as f:
    for line in test01:
        splitted_lines = line.split('\t')
        f.write(splitted_lines[1].strip() + "\n" )



# informal to formal
with open("pos2neg.txt", "r") as f:
    test10= f.readlines()

with open(directory +"/input_pos.txt","w") as f:
    for line in test10:
        splitted_lines = line.split('\t')
        f.write(splitted_lines[0].strip() + "\n")

with open(directory +"/output_neg.txt","w") as f:
    for line in test10:
        splitted_lines = line.split('\t')
        f.write(splitted_lines[1].strip() + "\n" )




