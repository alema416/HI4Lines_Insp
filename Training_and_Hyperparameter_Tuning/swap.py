import os
import shutil

def swap_floats_in_line(line):
    parts = line.strip().split()
    if len(parts) != 6:
        return line  # skip lines that do not match expected format

    # Swap float1 (index 2) and float4 (index 5)
    new_order = [parts[0], parts[5], parts[1], parts[2], parts[3], parts[4]]
    return ' '.join(new_order) + '\n'


def process_files_in_directory(directory):
    directory1 = './inference_tests/edge_server_train_swapped/labels'
    for filename in os.listdir(directory):
        print(filename)
        if filename.endswith('.txt'):
            filepath = os.path.join(directory, filename)
            filepath1 = os.path.join(directory1, filename)
            with open(filepath, 'r') as file:
                lines = file.readlines()

            swapped_lines = [swap_floats_in_line(line) for line in lines]

            with open(filepath1, 'w') as file:
                file.writelines(swapped_lines)

# Replace '.' with your directory path if different
process_files_in_directory('./inference_tests/edge_server_train/labels/')
#process_files_in_directory('./inference_tests/edge_server_val/labels/')