from metrics import evaluate


def load_input_data(file_paths):
    """Loads input sentences from given file paths.
    
    Args:
        file_paths (list): List containing file paths for input data.
    
    Returns:
        list: A list containing lists of sentences loaded from each file.
    """
    return [open(file_path, 'r').readlines() for file_path in file_paths]

def evaluate_model(input_data, output_dir, model_name, lambda_, model_type):
    """Evaluates a single model or sub-models for multi-output models.
    
    Args:
        input_data (list): List of input sentences for evaluation.
        output_dir (str): Directory path where model outputs are stored.
        model_name (str): Name of the model being evaluated.
        lambda_ (float): Lambda value used in evaluation.
        model_type (str): Type of the model ('single' or 'multi').
    
    Details:
        For single-output models, directly evaluates the model.
        For multi-output models, iterates through sub-models for evaluation.
    """
    sub_models = {
        'NAST_Huang': ["latentseq_learnable", "latentseq_simple", "stytrans_learnable", "stytrans_simple"],
        'StyTrans_Dai': ["condition", "multi"],
        'StyIns_Yi': ["2.5k", "10k", "unsuper"]
    }
    
    if model_type == 'single':
        model_list = [model_name]
    else:
        model_list = sub_models[model_name]
    
    for sub_model in model_list:
        prefix = f"{model_name}/{sub_model}/" if model_type == 'multi' else f"{model_name}/"
        output_files = [f"output_pos.txt", f"output_neg.txt"] if 'YELP' in output_dir else [f"output_formal.txt", f"output_informal.txt"]
        output_data = [open(f"{output_dir}/{prefix}{file_name}", 'r').readlines() for file_name in output_files]
        
        evaluations = [evaluate("yelp_0", input_data[1], output_data[1], lambda_), evaluate("yelp_1", input_data[0], output_data[0], lambda_)] if 'YELP' in output_dir \
                      else [evaluate("informal", input_data[0], output_data[1], lambda_), evaluate("formal", input_data[1], output_data[0], lambda_)]
        
        ret_mean = [(eval1 + eval2) / 2 for eval1, eval2 in zip(*evaluations)]
        print_evaluation_results(model_name if model_type == 'single' else f"{model_name}; {sub_model}", ret_mean)

def print_evaluation_results(model, metrics):
    """Prints the mean evaluation metrics for a given model.
    
    Args:
        model (str): The model name or identifier.
        metrics (list): List of mean metrics for both directions.
    """
    print(f"------------------- MEAN IN BOTH DIRECTIONS -------------------\n{model}")
    headers = '| ACC | COS | FL | J3 | J2 | mean3 | mean2 | g2 | h2 |'
    metrics_str = '|{:.3f}|' * len(metrics)
    print(f"{headers}\n{metrics_str.format(*metrics)}")

def evaluate_datasets():
    """Main function to evaluate all datasets."""

    DIR_input = {
        'YELP': ["data/yelp_0/test.txt", "data/yelp_1/test.txt"],
        'GYAFC': ["data/informal/test.txt", "data/formal/test.txt"]
    }
    
    DIR_output = {
        'YELP': "data/og/YELP",
        'GYAFC': "data/og/GYAFC"
    }
    
    lambda_values = {'YELP': 0.166, 'GYAFC': 0.115}
    
    models = {
        'YELP': {
            'single': ["deep_latent_seq_He", "DualRL_Luo", "Styins_Yi"],
            'multi': ["NAST_Huang", "StyTrans_Dai"]
        },
        'GYAFC': {
            'single': ["DualRL_Luo"],
            'multi': ["NAST_Huang", "StyIns_Yi"]
        }
    }
    
    for dataset, paths in DIR_input.items():
        input_data = load_input_data(paths)
        for model_type, model_names in models[dataset].items():
            for model_name in model_names:
                evaluate_model(input_data, DIR_output[dataset], model_name, lambda_values[dataset], model_type)

if __name__ == "__main__":
    evaluate_datasets()
