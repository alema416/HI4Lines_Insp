import os
import shutil
import json
from PIL import Image
import random
from tqdm import tqdm
import csv
from hydra import initialize, compose

random.seed(42)
with initialize(config_path="../configs/"):
    cfg = compose(config_name="data")  # exp1.yaml with defaults key

SCENE_WIDTH = cfg.data.scene_width #1280
CLASSIFIER_WIDTH = cfg.data.classifier_width #224

def search_csv(filename, bbox):
    with open("labels_filenames.csv", "r", newline="") as csvfile:
        reader = csv.reader(csvfile)
        
        for row in reader:
            if row[:5] == [filename] + list(map(str, bbox)):  # Convert ints to strings for comparison
                return row[5]
        else:
            print("No matching row found.")
            return 'ERROR'

def count_lines_in_txt(file_path):
    """Count the number of lines in a .txt file."""
    with open(file_path, 'r') as f:
        return sum(1 for line in f)

def find_matching_files(filename, dir_path):
    """Find files that start with the given filename in the specified directory."""
    matching_files = []
    for file in os.listdir(dir_path):
        if file.startswith(f'{filename}_'):
            matching_files.append(file)
    return matching_files

def compare_files_in_directories(dir_a, dir_b, dir_c):
    """Compare line count in .txt files with the number of matching files in other directories."""
    # Get list of .txt files in directory A
    txt_files_in_a = [f for f in os.listdir(dir_a) if f.endswith('.txt')]
    
    for txt_file in txt_files_in_a:
        txt_file_path = os.path.join(dir_a, txt_file)
        
        # Count lines in the txt file in directory A
        num_lines_in_txt = count_lines_in_txt(txt_file_path)
        
        # Get the filename without the .txt extension
        base_filename = os.path.splitext(txt_file)[0]
        
        # Look for matching files in directory B
        matching_files_in_b = find_matching_files(base_filename, dir_b)
        
        # If no matches in directory B, check directory C
        #if not matching_files_in_b:
        matching_files_in_c = find_matching_files(base_filename, dir_c)
        matching_files_in_b = matching_files_in_b + matching_files_in_c  # Combine matches from B and C

        # Compare the counts
        num_matching_files = len(matching_files_in_b)
        if num_lines_in_txt != num_matching_files:
            print('mismatch')
# Function to convert bounding box to YOLO format
def convert_to_yolo_format(CLASS_ID, bbox, original_width, original_height, resized_width, resized_height):
    x_min, y_min, box_width, box_height = bbox

    # Normalize the bounding box coordinates based on the original image size
    x_center = (x_min + (box_width / 2)) / original_width
    y_center = (y_min + (box_height / 2)) / original_height
    width = box_width / original_width
    height = box_height / original_height

    # Return normalized YOLO coordinates (center, width, height) based on resized image dimensions
    return [CLASS_ID, x_center, y_center, width, height]  # 0 is the class id for "insulator"

# Function to write the YOLO label to file
def write_yolo_label(label_file_path, yolo_data):
    with open(label_file_path, 'w') as f:
        for data in yolo_data:
            f.write(" ".join(map(str, data)) + '\n')

# Function to get image size
def get_image_size(image_path):
    with Image.open(image_path) as img:
        return img.size  # returns (width, height)
# Function to get image size

        
# Function to resize and save image while keeping aspect ratio
def resize_and_save_image(original_image_path, target_image_path, new_width):
    with Image.open(original_image_path) as img:
        original_width, original_height = img.size
        aspect_ratio = original_height / original_width
        new_height = int(new_width * aspect_ratio)
        resized_img = img.resize((new_width, new_height))
        resized_img.save(target_image_path)
        return original_width, original_height, new_width, new_height  # Return original and resized dimensions
        
def crop_image(original_image_path, bbox):
    
    with Image.open(original_image_path) as img:
        x_min, y_min, box_width, box_height = bbox
        cropped_img = img.crop((x_min, y_min, x_min + box_width, y_min + box_height))
        if cropped_img.height > cropped_img.width:
            new_width = CLASSIFIER_WIDTH
            aspect_ratio = cropped_img.height / cropped_img.width
            new_height = int(new_width * aspect_ratio)
            resized_cropped_img = cropped_img.resize((new_width, new_height))
        else:
            new_height = CLASSIFIER_WIDTH
            aspect_ratio = cropped_img.height / cropped_img.width
            new_width = int(new_height / aspect_ratio)
            resized_cropped_img = cropped_img.resize((new_width, new_height))
        
        return resized_cropped_img
# Function to process the data
def process_data_det(train_data, val_data, test_data, og_image_dir, yolo_m1_dir):


    # Define paths for train, val, test splits
    splits = {
        'train': {'image_dir': os.path.join(yolo_m1_dir, 'train/images'), 'label_dir': os.path.join(yolo_m1_dir, 'train/labels')},
        'val': {'image_dir': os.path.join(yolo_m1_dir, 'val/images'), 'label_dir': os.path.join(yolo_m1_dir, 'val/labels')},
        'test': {'image_dir': os.path.join(yolo_m1_dir, 'test/images'), 'label_dir': os.path.join(yolo_m1_dir, 'test/labels')}
    }

    # Create the directories if they don't exist
    for split in splits.values():
        os.makedirs(split['image_dir'], exist_ok=True)
        os.makedirs(split['label_dir'], exist_ok=True)

    # Function to process data for each split (train/val/test)
    def process_split(data, split_name):
        total_images_split = len(data)
        counter = 0
        #for entry in data:
        for entry in tqdm(data, desc=f'converting {split_name}', unit="img"):
            counter += 1
            #if counter % 500:
            #    print(f'{split_name}: {(counter / total_images_split)*100}% completed')
            filename = entry["filename"]
            labels = entry["Labels"]["objects"]

            # Original image path
            original_image_path = os.path.join(og_image_dir, filename)
            #print(filename)
            if not os.path.exists(original_image_path):
                print(f"Original image {filename} not found in {og_image_dir}. Skipping.")
                continue
            # Resize the image and save to the appropriate directory
            resized_image_path = os.path.join(splits[split_name]['image_dir'], filename)
            original_width, original_height, resized_width, resized_height = resize_and_save_image(original_image_path, resized_image_path, new_width=SCENE_WIDTH)

            yolo_data = []

            # Process each object, keep only the ones with string = 1
            for label in labels:
                if label["string"] == 1:
                    bbox = label["bbox"]
                    #print(f'insulator at: {bbox}')
                    
                    health_status = search_csv(filename, bbox)
                    #print(f'ins is: {health_status}')
                    if health_status == 'ERROR':
                        print('ERROR')
                    CLASS_ID = 0 if health_status == 'broken' else 1
                    yolo_format = convert_to_yolo_format(CLASS_ID, bbox, original_width, original_height, resized_width, resized_height)
                    yolo_data.append(yolo_format)

            if yolo_data:
                # Create the label file in the appropriate directory
                label_file_path = os.path.join(splits[split_name]['label_dir'], f"{os.path.splitext(filename)[0]}.txt")
                write_yolo_label(label_file_path, yolo_data)

    # Process each split
    process_split(train_data, 'train')
    process_split(val_data, 'val')
    process_split(test_data, 'test')

def process_data_class(train_data, val_data, test_data, og_image_dir, yolo_m1_dir):


    # Define paths for train, val, test splits
    splits = {
        'train': {'image_dir': os.path.join(yolo_m1_dir, 'train'), 'label_dir': os.path.join(yolo_m1_dir, 'train/labels')},
        'val': {'image_dir': os.path.join(yolo_m1_dir, 'val'), 'label_dir': os.path.join(yolo_m1_dir, 'val/labels')},
        'test': {'image_dir': os.path.join(yolo_m1_dir, 'test'), 'label_dir': os.path.join(yolo_m1_dir, 'test/labels')}
    }

    # Create the directories if they don't exist
    for split in splits.values():
        os.makedirs(split['image_dir'], exist_ok=True)
    for split in splits.values():
        os.makedirs(os.path.join(split['image_dir'], 'broken'), exist_ok=True)
        os.makedirs(os.path.join(split['image_dir'], 'healthy'), exist_ok=True)
    # Function to process each split (train/val/test)
    def process_split(data, split_name):
        image_id = 0
        total_images_split = len(data)
        counter = 0
        #for i in tqdm(range(total_steps), desc="Processing", unit="step"):
        for entry in tqdm(data, desc=f'converting {split_name}', unit="img"):
            #for entry in data:
            counter += 1
            #if counter % 500:
            #    print(f'{split_name}: {(counter / total_images_split)*100}% completed')        
            filename = entry["filename"]
            labels = entry["Labels"]["objects"]
            #print(filename)
            # Original image path
            original_image_path = os.path.join(og_image_dir, filename)

            if not os.path.exists(original_image_path):
                print(f"Original image {filename} not found in {og_image_dir}. Skipping.")
                continue

            # Get the original image size
            original_width, original_height = get_image_size(original_image_path)

            # Find all string=1 objects (the ones we will crop)
            string_1_objects = [label for label in labels if label["string"] == 1]
            for string_1 in string_1_objects:
                bbox_1 = string_1["bbox"]
                #print(f'insulator at: {bbox_1}')
                # Crop the image to this bounding box
                cropped_img = crop_image(original_image_path, bbox_1)
                cropped_width, cropped_height = cropped_img.size
                
                # Offset for repositioning string=0 bounding boxes
                offset_x, offset_y = bbox_1[0], bbox_1[1]
                
                # Save the cropped image
                cropped_image_filename = f"{os.path.splitext(filename)[0]}_crop_{image_id}.jpg"
                

                # Prepare YOLO labels for this cropped image
                yolo_data = []
                cropped_image_path = os.path.join(splits[split_name]['image_dir'], 'healthy', cropped_image_filename)
                # Process string=0 objects that are inside the string=1 bounding box
                label_for_filename = 'healthy'
                for label in labels:
                    
                    if label["string"] == 0:
                        bbox_0 = label["bbox"]
                        x_min_0, y_min_0, box_width_0, box_height_0 = bbox_0
                        
                        # Check if the string=0 bounding box is completely inside the string=1 bounding box
                        if x_min_0 >= bbox_1[0] and y_min_0 >= bbox_1[1] and (x_min_0 + box_width_0) <= (bbox_1[0] + bbox_1[2]) and (y_min_0 + box_height_0) <= (bbox_1[1] + bbox_1[3]):
                            #print(f'obj inside ins at: {bbox_0}')
                            if list(label['conditions'].values())[0] == 'Broken':
                                #print(f'{bbox_1} ins is broken')
                                label_for_filename = 'broken'
                                cropped_image_path = os.path.join(splits[split_name]['image_dir'], 'broken', cropped_image_filename)
                cropped_img.save(cropped_image_path)                  
                with open("labels_filenames.csv", "a", newline="") as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow([filename] + bbox_1 + [label_for_filename])
                image_id += 1

    # Process each split
    process_split(train_data, 'train')
    process_split(val_data, 'val')
    process_split(test_data, 'test')

# Main function
def main():
    modal = SCENE_WIDTH
    base_dir_og = cfg.data.base_dir_og
    json_file_path = os.path.join(base_dir_og, 'labels_v1.2.json')   
    og_image_dir = os.path.join(base_dir_og, 'Images')  
    
    yolo_m1_dir = cfg.data.base_dir_m1 #f'../data/processed/yolo_m1_JOIN_{modal}_debug'
    yolo_m2_dir = cfg.data.base_dir_m2 #f'../data/processed/IDID_cropped_224'  
    
    os.makedirs(yolo_m1_dir, exist_ok=True)
    os.makedirs(yolo_m2_dir, exist_ok=True)

    with open(json_file_path, 'r') as f:
        image_data = json.load(f)
    total_images = len(image_data)

    random.shuffle(image_data)

    train_data = image_data[:int(total_images * 0.64)]
    val_data = image_data[int(total_images * 0.64):int(total_images * 0.8)]
    test_data = image_data[int(total_images * 0.8):]    
    
    process_data_class(train_data, val_data, test_data, og_image_dir, yolo_m2_dir)
    print("M2 Processing complete. Resized images and YOLO labels have been saved.")
    process_data_det(train_data, val_data, test_data, og_image_dir, yolo_m1_dir)
    print("M1 processing complete. Resized images and YOLO labels have been saved.")
    
    for mode in ['train', 'val', 'test']:
        print(f"M1 {mode}: {len(os.listdir(os.path.join(yolo_m1_dir, mode, 'images')))}")
        print(f"M2 {mode}: {len(os.listdir(os.path.join(yolo_m2_dir, mode, 'broken'))) + len(os.listdir(os.path.join(yolo_m2_dir, mode, 'healthy')))} ({len(os.listdir(os.path.join(yolo_m2_dir, mode, 'healthy')))} + {len(os.listdir(os.path.join(yolo_m2_dir, mode, 'broken')))})")
    
    for mode in ['train', 'val', 'test']:
        dir_a = os.path.join(yolo_m1_dir, f"{mode}/labels/")
        dir_b = os.path.join(yolo_m2_dir, f"{mode}/broken/")
        dir_c = os.path.join(yolo_m2_dir, f"{mode}/healthy/")

        compare_files_in_directories(dir_a, dir_b, dir_c)
    print('check completed')
if __name__ == "__main__":
    main()
