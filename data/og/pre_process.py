import os

def create_directory(directory_name):
    """Create a new directory if it does not already exist."""
    if not os.path.isdir(directory_name):
        os.mkdir(directory_name)

def process_and_save(file_path, input_file, output_file):
    """
    Process a file to separate inputs and outputs, then save them to specified files.
    
    Args:
        file_path (str): Path to the file to be processed.
        input_file (str): File path to save the input part.
        output_file (str): File path to save the output part.
    """
    with open(file_path, "r") as f:
        lines = f.readlines()

    inputs = [line.split('\t')[0].strip() for line in lines]
    outputs = [line.split('\t')[1].strip() for line in lines]

    with open(input_file, "w") as f:
        f.write("\n".join(inputs) + "\n")

    with open(output_file, "w") as f:
        f.write("\n".join(outputs) + "\n")

if __name__ == "__main__":
    directory = "seperated_in_out"
    create_directory(directory)

    # Formal to Informal
    process_and_save("neg2pos.txt", os.path.join(directory, "input_neg.txt"), os.path.join(directory, "output_pos.txt"))
    
    # Informal to Formal
    process_and_save("pos2neg.txt", os.path.join(directory, "input_pos.txt"), os.path.join(directory, "output_neg.txt"))
