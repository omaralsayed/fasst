# FASST: Few-Shot Abstractive Summarization for Style Transfer

Codebase for [FASST](https://ieeexplore.ieee.org/abstract/document/10236629).

## Setup
### Dependencies
```
pip install -r requirements.txt
```

### Data
Although the model is written to work with any input dataset, we provide instructions to obtain and setup the Yelp and GYAFC datasets as used to evaluate the model in the paper.

If using different datasets, modify `data/paths.json` to match your structure.

Under the `data` directory, create directories named after each style in the datasets. For example, `yelp_0` and `yelp_1`. Into each directory, place the corresponding `dev.txt`, `test.txt` and `train.txt` files containing all of the data for those splits with one entry per line.

### Yelp Positive/Negative Data
The Yelp data is [provided publicly](https://github.com/lijuncen/Sentiment-and-Style-Transfer/tree/master/data/yelp). We take `sentiment.0` and `sentiment.1` as `yelp_0` and `yelp_1`, corresponding to the negative sentiment and positive sentiment reviews, respectively. The files can be downloaded from the public repository, placed in their corresponding directory, and renamed according to the split they represent. Ensure that the data matches the corresponding entries in `data/paths.json`.

### GYAFC Formal/Informal Data
Grammarly's Yahoo Answers Formality Corpus (GYAFC) is available at no cost to researchers, but access must be requested from the dataset authors. Please see the [instructions here](https://github.com/raosudha89/GYAFC-corpus). Once access is granted and the dataset is obtained, the data can be taken as is and organized into the structure outlined above. In the model evaluation, we only take the Family & Relationships domain as our formality dataset.

### Creating Embeddings
Once the data is in place, it needs to be embedded and saved. We provide `embed.py` to perform this task. The following command will embed and save all of the splits for each style in `data/paths.json`:
```
python embed.py
```

## Running FASST
After completing the above setup, placing the data and CoLA model, and creating the embeddings, the model can be ran using the simplified pipeline.

Example command for running FASST to transfer the Yelp positive (1) test split to negative (0) sentiment:
```
python fasst.py --input ./data/yelp_1/test.txt --source_style yelp_1 --target_style yelp_0
```

Command line arguments for running are as follows:
```
--k [int]    default: 4    Objects (nearest neighbors)
--r [int]    default: 3    Subset size to summarize
--input [str]    Input file
--source_style [str]    Source style
--target_style [str]    Target style
--lambda_score [str]    Lambda score
```

## Running the Metrics
We provide three files to run metrics for models we evaluate.

The file `metric/fasst_out.py` is used to calculate the metrics for FASST model output.
```
python metric/fasst_out.py
```

`metric/luo_out.py` can be used to calculate the metrics for the output data included in `data/luo/`.
```
python metric/luo_out.py
```

`metric/og_out.py` can be used to calculate the metrics for the output data included in `data/og/`.
```
python metric/og_out.py
```

## FASST Outputs
Text files containing our model outputs for Yelp and GYAFC are available in the `data/output/` directory. The files are labeled as according to the source followed by target style, then the parameters used in the re-ranking process outlined in the paper.
