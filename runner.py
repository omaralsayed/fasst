import os
import gc

import torch

def clean():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

clean()

# k = 5, r = 3

# os.system("python fasst.py --input ./data/informal/test.txt --source_style informal --target_style formal --k 5 --r 3 --lambda_score 0.03")
# clean()
os.system("python fasst.py --input ./data/formal/test.txt --source_style formal --target_style informal --k 5 --r 3 --lambda_score 0.03")
clean()
# os.system("python fasst.py --input ./data/yelp_0/test.txt --source_style yelp_0 --target_style yelp_1 --k 5 --r 3 --lambda_score 0.15")
# clean()
# os.system("python fasst.py --input ./data/yelp_1/test.txt --source_style yelp_1 --target_style yelp_0 --k 5 --r 3 --lambda_score 0.15")
# clean()

# k = 4, r = 3

# os.system("python fasst.py --input ./data/informal/test.txt --source_style informal --target_style formal --k 4 --r 3 --lambda_score 0.03")
# clean()
os.system("python fasst.py --input ./data/formal/test.txt --source_style formal --target_style informal --k 4 --r 3 --lambda_score 0.03")
clean()
# os.system("python fasst.py --input ./data/yelp_0/test.txt --source_style yelp_0 --target_style yelp_1 --k 4 --r 3 --lambda_score 0.15")
# clean()
# os.system("python fasst.py --input ./data/yelp_1/test.txt --source_style yelp_1 --target_style yelp_0 --k 4 --r 3 --lambda_score 0.15")
# clean()

# k = 5, r = 4

# os.system("python fasst.py --input ./data/informal/test.txt --source_style informal --target_style formal --k 5 --r 4 --lambda_score 0.03")
# clean()
os.system("python fasst.py --input ./data/formal/test.txt --source_style formal --target_style informal --k 5 --r 4 --lambda_score 0.03")
clean()
# os.system("python fasst.py --input ./data/yelp_0/test.txt --source_style yelp_0 --target_style yelp_1 --k 5 --r 4 --lambda_score 0.15")
# clean()
# os.system("python fasst.py --input ./data/yelp_1/test.txt --source_style yelp_1 --target_style yelp_0 --k 5 --r 4 --lambda_score 0.15")
# clean()

# Base case (k=3)

# os.system("python fasst.py --input ./data/informal/test.txt --source_style informal --target_style formal --k 3 --r 3 --lambda_score 0.03")
# clean()
os.system("python fasst.py --input ./data/formal/test.txt --source_style formal --target_style informal --k 3 --r 3 --lambda_score 0.03")
clean()
# os.system("python fasst.py --input ./data/yelp_0/test.txt --source_style yelp_0 --target_style yelp_1 --k 3 --r 3 --lambda_score 0.15")
# clean()
# os.system("python fasst.py --input ./data/yelp_1/test.txt --source_style yelp_1 --target_style yelp_0 --k 3 --r 3 --lambda_score 0.15")
# clean()

# Base case (k=4)

# os.system("python fasst.py --input ./data/informal/test.txt --source_style informal --target_style formal --k 4 --r 4 --lambda_score 0.03")
# clean()
os.system("python fasst.py --input ./data/formal/test.txt --source_style formal --target_style informal --k 4 --r 4 --lambda_score 0.03")
clean()
# os.system("python fasst.py --input ./data/yelp_0/test.txt --source_style yelp_0 --target_style yelp_1 --k 4 --r 4 --lambda_score 0.15")
# clean()
os.system("python fasst.py --input ./data/yelp_1/test.txt --source_style yelp_1 --target_style yelp_0 --k 4 --r 4 --lambda_score 0.15")
clean()