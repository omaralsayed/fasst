import os


def run_fasst_transformations():
    configurations = [
        {'k': 5, 'r': 3, 'lambda_score_informal': 0.03, 'lambda_score_yelp': 0.15},
        {'k': 4, 'r': 3, 'lambda_score_informal': 0.03, 'lambda_score_yelp': 0.15},
        {'k': 5, 'r': 4, 'lambda_score_informal': 0.03, 'lambda_score_yelp': 0.15},
        {'k': 3, 'r': 3, 'lambda_score_informal': 0.03, 'lambda_score_yelp': 0.15},
        {'k': 4, 'r': 4, 'lambda_score_informal': 0.03, 'lambda_score_yelp': 0.15},
    ]

    data_paths = [
        ('./data/informal/test.txt', 'informal', 'formal'),
        ('./data/formal/test.txt', 'formal', 'informal'),
        ('./data/yelp_0/test.txt', 'yelp_0', 'yelp_1'),
        ('./data/yelp_1/test.txt', 'yelp_1', 'yelp_0'),
    ]

    for config in configurations:
        for data_path, source_style, target_style in data_paths:
            lambda_score = config['lambda_score_informal'] if 'informal' in data_path or 'formal' in data_path else config['lambda_score_yelp']
            command = f"python fasst.py --input {data_path} --source_style {source_style} --target_style {target_style} --k {config['k']} --r {config['r']} --lambda_score {lambda_score}"
            os.system(command)

if __name__ == "__main__":
    run_fasst_transformations()
