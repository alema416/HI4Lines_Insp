import os
import shutil

def copy_matching_files(dir_a, dir_b, dir_c):
    """
    For each file in directory A, copies the file with the same filename from directory B to directory C.
    
    Parameters:
    - dir_a (str): Path to the directory A (source of filenames).
    - dir_b (str): Path to the directory B (source of files to copy).
    - dir_c (str): Path to the directory C (destination directory).
    """
    # Ensure the destination directory exists
    os.makedirs(dir_c, exist_ok=True)

    # Iterate over each file in directory A
    for filename in os.listdir(dir_a):
        # Construct the full path for the file in A
        file_a_path = os.path.join(dir_a, filename)
        # Only consider files (skip subdirectories)
        if os.path.isfile(file_a_path):
            # Construct the expected source file path in directory B
            file_b_path = os.path.join(dir_b, filename)
            # Check if the file exists in directory B
            if os.path.isfile(file_b_path):
                # Define the destination file path in directory C
                file_c_path = os.path.join(dir_c, filename)
                # Copy the file from B to C
                shutil.copy(file_b_path, file_c_path)
                print(f"Copied {filename} from {dir_b} to {dir_c}.")
            else:
                print(f"File '{filename}' not found in {dir_b}.")


GT = '/home/amax/machairas/FMFP-edge-idid/yolo_m1_JOIN_1280_debug/'
for thres in np.arange(0.5, 1.01, 0.05):
    thres = f'{thres:.2f}'
    for split in ['train', 'val', 'test']:
        SERVER_INF = '/home/amax/machairas/inference_tests/edge_server_{split}_swapped/'
        copy_matching_files("./runs/threshold_{thres}_offloaders/{split}", os.path.join(GT, split, labels), "./offloaders_gt/{thres}/{split}")
        copy_matching_files("./runs/threshold_{thres}_offloaders/{split}", os.path.join(SERVER_INF, labels), "./offloaders_server_inf/{thres}/{split}")
        print(f'{thres} {split} done!')