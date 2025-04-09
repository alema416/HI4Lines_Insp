import json
import os
from PIL import Image

# Set the paths for the input and output directories
for gt in [True, False]:
    for split in ['val', 'test']:
        print(gt)
        #direct = input(f'{split}: ') if not gt else ''
        #direct = input(f'{split}: ')
        prediction_path = f"/home/amax/machairas/inference_tests/test_multi_{split}/labels/"
        multiclass = True
        input_dir = f'./yolo_m1_JOIN_1280_debug/{split}/images/'
        input_dir_anno = f'./yolo_m1_JOIN_1280_debug/{split}/labels/' if gt else prediction_path
        output_dir = './coco_dataset/'
        
        # Define the categories for the COCO dataset (is this ok?)
        categories = [{"id": 0, "name": "broken"}, {"id": 1, "name": "healthy"}]
        
        # Define the COCO dataset dictionary
        coco_dataset = {
            "info": {},
            "licenses": [],
            "categories": categories,
            "images": [],
            "annotations": []
        }
        counter = 1
        # Loop through the images in the input directory
        for image_file in os.listdir(input_dir):
            
            # Load the image and get its dimensions
            image_path = os.path.join(input_dir, image_file)
            image = Image.open(image_path)
            width, height = image.size
            
            # Add the image to the COCO dataset
            image_dict = {
                "id": counter, #int(image_file.split('.')[0]),
                "width": width,
                "height": height,
                "file_name": image_file
            }
            coco_dataset["images"].append(image_dict)
            
            # Load the bounding box annotations for the image
            with open(os.path.join(input_dir_anno, f'{image_file.split(".")[0]}.txt')) as f:
                annotations = f.readlines()
            
            # Loop through the annotations and add them to the COCO dataset
            for ann in annotations:
                cat_id, x, y, w, h = map(float, ann.strip().split())
                x_min, y_min = int((x - w / 2) * width), int((y - h / 2) * height)
                x_max, y_max = int((x + w / 2) * width), int((y + h / 2) * height)
                ann_dict = {
                    "id": len(coco_dataset["annotations"]),
                    "image_id": counter, #int(image_file.split('.')[0]),
                    "category_id": int(cat_id),
                    "bbox": [x_min, y_min, x_max - x_min, y_max - y_min],
                    "area": (x_max - x_min) * (y_max - y_min),
                    "iscrowd": 0
                }
                coco_dataset["annotations"].append(ann_dict)
            counter += 1
        # Save the COCO dataset to a JSON file
        with open(os.path.join(output_dir, f'annotations_{split}_multiclass_{gt}.json'), 'w') as f:
            json.dump(coco_dataset, f)
