from metrics import evaluate


DIR = {
    "YELP": "data/luo/YELP",
    "GYAFC": "data/luo/GYAFC"
}

def load_data(file_paths):
    """
    Load data from specified file paths.

    Args:
        file_paths (list): List of file paths to load data from.

    Returns:
        list: List of data loaded from the files.
    """
    data = []
    for file_path in file_paths:
        with open(file_path, "r") as file:
            data.append(file.readlines())
    return data

def evaluate_model(model, data_source, input_data, lambda_):
    """
    Evaluate a given model using specified input data and lambda value.

    Args:
        model (str): Name of the model to evaluate.
        data_source (str): Source of data ('YELP' or 'GYAFC').
        input_data (list): List of input data for evaluation.
        lambda_ (float): Lambda value for evaluation.

    Returns:
        list: List of mean metrics from the evaluation.
    """
    print(data_source.upper())
    output_files = {
        "YELP": ["output_pos.txt", "output_neg.txt"],
        "GYAFC": ["output_formal.txt", "output_informal.txt"]
    }
    target_styles = {
        "YELP": ["yelp_0", "yelp_1"],
        "GYAFC": ["informal", "formal"]
    }

    output_data = []
    for output_file in output_files[data_source]:
        file_path = f"{DIR[data_source]}/{model}/{output_file}"
        with open(file_path, "r") as file:
            output_data.append(file.readlines())

    evaluations = [evaluate(target_style, input_set, output_set, lambda_) 
                   for target_style, input_set, output_set in zip(target_styles[data_source], input_data, output_data)]

    # Calculate and print the mean in both directions
    ret_mean = [(evaluations[0][i] + evaluations[1][i]) / 2 for i in range(len(evaluations[0]))]
    return ret_mean

def print_metrics(metrics, model):
    """
    Print the evaluation metrics.

    Args:
        metrics (list): List of metrics to print.
        model (str): Name of the model being evaluated.
    """
    print("------------------- MEAN IN BOTH DIRECTIONS -------------------")
    print(f"{model}")
    headers = "| ACC | COS | FL |  J3 |  J2 | mean3| mean2| g2 | h2 |"
    separators = "| --- | --- | -- | --- | --- | ---- | ---- | -- | -- |"
    metrics_str = '|{:.3f}|{:.3f}|{:.3f}|{:.3f}|{:.3f}|{:.3f}|{:.3f}|{:.3f}|{:.3f}|\n'.format(*metrics)
    print(f"{headers}\n{separators}\n{metrics_str}")

def run_metrics(data="YELP", lambda_=0.166):
    """
    Run metrics for specified data source and lambda value.

    Args:
        data (str): Data source ('YELP' or 'GYAFC').
        lambda_ (float): Lambda value for evaluation.
    """
    file_paths = {
        "YELP": [DIR_input_neg, DIR_input_pos],
        "GYAFC": [DIR_input_informal, DIR_input_formal]
    }

    input_data = load_data(file_paths[data])

    model_list = ["StyleEmbedding_Fu"]  # Using as an example.
    for model in model_list:
        metrics = evaluate_model(model, data, input_data, lambda_)
        print_metrics(metrics, model)

if __name__ == "__main__":
    DIR_input_neg = "data/yelp_0/test.txt"
    DIR_input_pos = "data/yelp_1/test.txt"
    DIR_input_formal = "../GYAFC_test/formal.txt"
    DIR_input_informal = "../GYAFC_test/informal.txt"

    run_metrics(data="YELP", lambda_=0.166)
    run_metrics(data="GYAFC", lambda_=0.115)
