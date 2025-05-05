import degirum as dg, degirum_tools
import time
import os
import argparse
import logging
from tqdm import tqdm 
from PIL import Image
import numpy as np
def convert_bbox_to_yolo(classid, score, x1, y1, x2, y2, img_width, img_height):
    bbox_width = (x2 - x1) / img_width
    bbox_height = (y2 - y1) / img_height
    x_center = (x1 + x2) / 2 / img_width
    y_center = (y1 + y2) / 2 / img_height

    return classid, score, x_center, y_center, bbox_width, bbox_height
        
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")
logging.getLogger("degirum").setLevel(logging.WARNING)

parser = argparse.ArgumentParser(description='Rethinking CC for FP')
parser.add_argument('--THRES', default=0.5, type=float, help='')
parser.add_argument('--SPLIT', default='val', type=str, help='')
args = parser.parse_args()

# choose inference host address
inference_host_address = "@local"
# inference_host_address = "@local"

# choose zoo_url
zoo_url1 = "./models/"
zoo_url2 = "./models/" #"degirum/hailo"
# zoo_url = "../models"

# set token
token = 'dg_BdpiWu9QKDruFarDuvabz8otbWpRWNFvFK9DV'
# token = '' # leave empty for local inference

face_det_model_name = 'yolov11n_vanilla_simplified' #"yolov8n_relu6_face--640x640_quant_hailort_hailo8_1"
gender_cls_model_name = 'resnet_v1_18_custom_code' #"yolov8n_relu6_fairface_gender--256x256_quant_hailort_hailo8_1"
video_source = "./assets/output.mp4"


# Load face detection and gender detection models
face_det_model = dg.load_model(
    model_name=face_det_model_name,
    inference_host_address=inference_host_address,
    zoo_url=zoo_url1,
    token=token,
    overlay_color=[(255,255,0),(0,255,0)]    
)

gender_cls_model = dg.load_model(
    model_name=gender_cls_model_name,
    inference_host_address=inference_host_address,
    zoo_url=zoo_url2,
    token=token,
)

# Create a compound cropping model with 20% crop extent
crop_model = degirum_tools.CroppingAndClassifyingCompoundModel(
    face_det_model, 
    gender_cls_model, 
    30.0
)
# run inference on images ( for global evaluation )

THRES = args.THRES
offloaded = 0
total_images = 0
times = []

split = args.SPLIT
os.makedirs(f'./runs/threshold_{THRES:.2f}/{split}', exist_ok=True)
os.makedirs(f'./runs/threshold_{THRES:.2f}_offloaders/{split}', exist_ok=True)
offloaders = []
path = f"/home/amax/GitHub/hailo_examples/assets/detection_{split}_set/images/"
for image in tqdm(os.listdir(path)):
    start_time = time.time()    
    
    im = Image.open(os.path.join(path, image))
    width, height = im.size
    
    total_images += 1
    try:
        inference_results = crop_model.predict(os.path.join(path, image)) # fix resize bug
    except Exception as e:
        logging.error(os.path.join(path, image))
        logging.error(str(e))
        #print(f'{width}, {height}')
        #print(inference_results)
    lines = [] 
    status = 1
    for i in inference_results.results:
        #logging.debug()
        lines.append(convert_bbox_to_yolo(i['category_id'], i['score'], i['bbox'][0], i['bbox'][1],i['bbox'][2], i['bbox'][3], width, height))
        if i['score'] < THRES:
            status = 0
    if status:
        #print(lines)
        with open(f'./runs/threshold_{THRES:.2f}/{split}/{os.path.splitext(image)[0]}.txt', 'w') as file:
            for line in lines:
                #file.write(f"{line}\n")
                file.write(' '.join(map(str, line)) + '\n')
    else:
        offloaded += 1
        offloaders.append(f'{os.path.splitext(image)[0]}.txt')
        with open(f'./runs/threshold_{THRES:.2f}_offloaders/{split}/{os.path.splitext(image)[0]}.txt', 'w') as file:
            for line in lines:
                #file.write(f"{line}\n")
                file.write(' '.join(map(str, line)) + '\n')
    end_time = time.time()
    times.append(end_time - start_time)
    #print(end_time - start_time)

print(offloaders)
arr = np.array(times[1:])
mean = np.mean(arr)
std = np.std(arr)

print(f"Mean: {mean:.4f}")
print(f"Standard Deviation: {std:.4f}")
logging.info(f'{THRES}: offloaded {offloaded} out of {total_images} images')