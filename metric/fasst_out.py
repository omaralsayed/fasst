from metrics import evaluate


def read_files(input_path, output_path, preprocess_output=False):
    """Read input and output files, optionally preprocess outputs.

    Args:
        input_path (str): Path to the input file.
        output_path (str): Path to the output file.
        preprocess_output (bool): Flag to preprocess output lines.

    Returns:
        tuple: Tuple containing lists of inputs and outputs.
    """
    with open(input_path, "r") as input_file, open(output_path, "r", encoding="ISO-8859-1" if preprocess_output else "utf-8") as output_file:
        inputs = input_file.readlines()
        outputs = [line.replace("Input:", "").strip() if preprocess_output else line.strip() for line in output_file.readlines()]
    return inputs, outputs

def write_processed_outputs(outputs, path):
    """Write processed outputs to a file.

    Args:
        outputs (list): List of processed output strings.
        path (str): File path to write the outputs.
    """
    with open(path, "w") as file:
        for line in outputs:
            file.write(line + "\n")

def evaluate_and_print_metrics(target_style, inputs, outputs, lambda_, description):
    """Evaluate style transfer metrics and print the results.

    Args:
        target_style (str): Target style for evaluation.
        inputs (list): List of input sentences.
        outputs (list): List of output sentences.
        lambda_ (float): Lambda parameter for evaluation.
        description (str): Description of the evaluation context.
    """
    print("*" * 50)
    print(description)
    print("*" * 50)

    return evaluate(target_style, inputs, outputs, lambda_)

def calculate_and_print_mean_metrics(metrics1, metrics2, k, r):
    """Calculate and print mean metrics from two sets of evaluations.

    Args:
        metrics1 (list): First set of metrics.
        metrics2 (list): Second set of metrics.
        k (int): Parameter k in evaluation.
        r (int): Parameter r in evaluation.
    """
    mean_metrics = [(m1 + m2) / 2 for m1, m2 in zip(metrics1, metrics2)]
    print("------------------- MEAN IN BOTH DIRECTIONS -------------------")
    print(f"for k={k}, r={r}")
    print('| ACC | COS | FL |  J3  | J2 | mean3 | mean2 | g2 | h2 |\n')
    print('|' + '|'.join(f'{metric:.3f}' for metric in mean_metrics) + '|\n')

def process_dataset(DIR, combination, lambda_, target_styles, process_output_func):
    """Process a dataset with given parameters and target styles.

    Args:
        DIR (str): Base directory for the dataset.
        combination (list of tuples): List of (k, r) combinations to evaluate.
        lambda_ (float): Lambda parameter for evaluation.
        target_styles (dict): Dictionary with keys as 'input_to_output' and 'output_to_input' target styles.
        process_output_func (function): Function to process outputs before evaluation.
    """
    for k, r in combination:
        for direction, target_style in target_styles.items():
            input_path = f"{DIR}/{direction}_input_k{k}_r{r}.txt"
            output_path = f"{DIR}/{direction}_output_k{k}_r{r}.txt"
            inputs, outputs = read_files(input_path, output_path, preprocess_output=True if process_output_func else False)

            if process_output_func:
                process_output_func(outputs, f"data/output/{direction}_output_k{k}_r{r}.txt")

            description = f"{direction.replace('_', ' ')} for k={k}, r={r}, len={len(outputs)}"
            metrics = evaluate_and_print_metrics(target_style, inputs, outputs, lambda_, description)
            yield metrics

if __name__ == "__main__":
    DIR_YELP = "data/fasst/YELP"
    DIR_GYAFC = "data/fasst/GYAFC"
    
    lambda_GYAFC = 0.115
    lambda_YELP = 0.166

    combinations = [
        (3, 3),
        (4, 4),
        (4, 3),
        (5, 4)
    ]

    # GYAFC Processing
    target_styles_GYAFC = {
        "formal_to_informal": "informal",
        "informal_to_formal": "formal"
    }

    gyafc_metrics = list(process_dataset(DIR_GYAFC, combinations, lambda_GYAFC, target_styles_GYAFC, write_processed_outputs))

    # Yelp Processing
    target_styles_YELP = {
        "0_to_1": "yelp_1",
        "1_to_0": "yelp_0"
    }

    yelp_metrics = list(process_dataset(DIR_YELP, combinations, lambda_YELP, target_styles_YELP, write_processed_outputs))
