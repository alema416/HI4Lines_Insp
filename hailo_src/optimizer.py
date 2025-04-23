import json
import os
import csv
import argparse
import numpy as np
from nmetr import AUGRC
from nmetr import calc_aurc_eaurc
from nmetr import calc_fpr_aupr
import torch
import pandas as pd
import tensorflow as tf
from IPython.display import SVG
import matplotlib
from matplotlib import patches
from matplotlib import pyplot as plt
from PIL import Image
from tensorflow.python.eager.context import eager_mode
from hailo_sdk_client import ClientRunner, InferenceContext
matplotlib.use("Agg")

parser = argparse.ArgumentParser(description='Rethinking CC for FP')
parser.add_argument('--run_id', required=True, type=int, help='')
parser.add_argument('--pathh', required=True, type=str, help='')
args = parser.parse_args()


# -----------------------------------------
# CUSTOM (degirum used closed-source softmax
# -----------------------------------------
def stable_softmax(x):
    z = x - np.max(x)
    exp_scores = np.exp(z)
    return exp_scores / np.sum(exp_scores)

# -----------------------------------------
# Pre processing
# -----------------------------------------


def preproc(image, output_height=224, output_width=224, resize_side=224, normalize=False):
    with eager_mode():
        resized_image = tf.compat.v1.image.resize_bilinear(tf.expand_dims(image, 0), [output_height, output_width])
        cropped_image = tf.compat.v1.image.resize_with_crop_or_pad(resized_image, output_height, output_width)

        if normalize:
            cropped_image = \
                (cropped_image - [0.50463295*255, 0.46120012*255, 0.4291694*255 ]) / [0.18364702*255, 0.1885083*255,  0.19882548*255]

        return tf.squeeze(cropped_image)


# -----------------------------------------------------
# Post processing
# -----------------------------------------------------

def _get_idid_labels():
    example_names = ['broken', 'healthy']
    return example_names


idid_labels = _get_idid_labels()


def postproc(results):
    labels = []
    scores = []
    results = [np.squeeze(result) for result in results]
    
    for result in results:
        exp_scores = stable_softmax(result)
        probabilities = exp_scores / np.sum(exp_scores)
        top_ind = np.argmax(probabilities)
        cur_label = idid_labels[top_ind]
        cur_score = exp_scores[top_ind]
        labels.append(cur_label)
        scores.append(cur_score)
        
    return scores, labels

# -------------
# normalization (for matplotlib only)
# -------------
def mynorm(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

# -------------
# metrics calculation and per sample csv logging
# -------------

def visualize_results(images, images_path, split, scores=None, inf_labels=None, place=None):
    images_list = sorted([img_name for img_name in os.listdir(images_path) if os.path.splitext(img_name)[1] == ".jpg"])

    success_confidences = []
    error_confidences = []
    total_samples = 0
    total_correct = 0
    
    assert (scores is None and inf_labels is None) or (
        scores is not None and inf_labels is not None
    ), "scores and labels must both be supplied, or both not be supplied"


    assert len(images) == len(scores) == len(inf_labels), "lengths of inputs must be equal"


    accuracy = 0.
    list_correct = []
    list_softmax = []
    #list_logit = []
    labels = []
    with open(os.path.join('./', f'per_sample_{place}_{split}.csv'), 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
    # Write a header row (optional)
        csv_writer.writerow(['File Name', 'Predicted Label', \
                                'Ground Truth Label', 'Confidence (%)', 'softmax_max', 'softmax_min', 'Status', 'correct'])

        for img_idx in range(len(images)):
            total_samples += 1

            real_label = None
            if os.path.isfile(os.path.join(f'./yolo_m2_class_square_JOIN_224/{split}/broken', images_list[img_idx])):
                real_label = 'broken' 
            elif os.path.isfile(os.path.join(f'./yolo_m2_class_square_JOIN_224/{split}/healthy', images_list[img_idx])):
                real_label = 'healthy'
            else:
                print('==================ERROR==================')
            crr = None
            labels.append(real_label)
            
            conf_pct = scores[img_idx]
            if inf_labels[img_idx] == real_label:
                #total_correct += 1
                #success_confidences.append(conf_pct)
                accuracy += 1
                cor = 1
                csv_writer.writerow([images_list[img_idx], inf_labels[img_idx], real_label, f'{conf_pct:.2f}', 'no', 'no', 'Success', True])

            else:
                cor = 0
                csv_writer.writerow([images_list[img_idx], inf_labels[img_idx], real_label, f'{conf_pct:.2f}', 'no', 'no', 'Error', False])
                error_confidences.append(conf_pct)
            list_correct.append(cor)
            
            softmax_max = conf_pct
            softmax_min = 1 - conf_pct
            
            # ensure softmax correct order - not {max, min} but {broken, healthy}
            if inf_labels[img_idx] == 'broken':
                crr = 1
            else:
                crr = 0
            
            if crr:
                list_softmax.append([softmax_max, softmax_min])
            else:
                list_softmax.append([softmax_min, softmax_max])
        overall_accuracy = 100 * accuracy / len(list_correct)
        print(f'{accuracy} success, {len(list_correct) - accuracy} error')
    
    probs = torch.tensor(list_softmax, dtype=torch.float32)  # Now shape (N, C)
    label_map = {'broken': 0, 'healthy': 1}
    numeric_labels = [label_map[label] if label in label_map else -1 for label in labels]
    # If any labels are -1, you'll want to handle them accordingly.
    
    numeric_labels_tensor = torch.tensor(numeric_labels, dtype=torch.long)
    
    #print(f"First 5 numeric labels: {numeric_labels}")
    #print(f"First 5 predicted class indices: {probs.argmax(dim=-1)}")
    
    # Debug the tensor shapes
    #print(f"probs shape: {probs.shape}, numeric_labels_tensor shape: {numeric_labels_tensor.shape}")
    #print(f"probs last 5: {probs[:5]}")  # Print first 5 for sanity check
    #print(f"numeric_labels_tensor last 5: {numeric_labels_tensor[:5]}")  # Check labels
    #print(f'{split} accuracy: {overall_accuracy}')
    # Check if labels contain -1 (invalid)
    if -1 in numeric_labels:
        print("ERROR: Found -1 in numeric_labels. Check label mapping!")
    
    
    #print(f"Probs min: {probs.min()}, max: {probs.max()}")
    augrc_metric = AUGRC()

    # Update the AUGRC metric with the predicted probabilities and ground-truth labels
    augrc_metric.update(probs, numeric_labels_tensor)

    # Compute the AUGRC value
    augrc_value = augrc_metric.compute()
    
    res = calc_aurc_eaurc(list_softmax, list_correct)
    res2 = calc_fpr_aupr(list_softmax, list_correct)
    
    print(f'================== {place}, {split} accuracy: {overall_accuracy}==================')
    print(f'SPECIAL_PRINT{place}{split}acc {overall_accuracy}')
    print(f'SPECIAL_PRINT{place}{split}augrc {1000*augrc_value.item()}')
    return overall_accuracy, 100*res2[0], 100*res2[1], 100*res2[2], 100*res2[3], 100*res2[4], res[0]*1000, res[1]*1000, 1000*augrc_value.item()


# -------------
# load HAR file and prepare dataset
# -------------
har_dir = './'
onnx_dir = './'
quantized_har_dir = './'
model_name = f'model_{args.run_id}'
onnx_model_name = model_name

print(onnx_model_name)

hailo_model_har_name = os.path.join(har_dir, f"{onnx_model_name}_hailo_model.har")

assert os.path.isfile(hailo_model_har_name), "Please provide valid path for HAR file"
runner = ClientRunner(har=hailo_model_har_name)


images_path = './yolo_m2_class_square_JOIN_224/val/all'
images_list = [img_name for img_name in os.listdir(images_path) if os.path.splitext(img_name)[1] == ".jpg"]


image_dataset = np.zeros((len(images_list), 224, 224, 3))


image_dataset_normalized = np.zeros((len(images_list), 224, 224, 3))
for idx, img_name in enumerate(sorted(images_list)):
    img = np.array(Image.open(os.path.join(images_path, img_name)))
    img_preproc = preproc(img)
    image_dataset[idx, :, :, :] = img_preproc.numpy()
    img_preproc_norm = preproc(img, normalize=False)
    image_dataset_normalized[idx, :, :, :] = img_preproc_norm.numpy()

# -------------
# run native model inference on emulator
# -------------

with runner.infer_context(InferenceContext.SDK_NATIVE) as ctx:
    native_res = runner.infer(ctx, image_dataset_normalized[:, :, :, :])

native_scores, native_labels = postproc(native_res)
aaa = visualize_results(image_dataset[:, :, :, :], './yolo_m2_class_square_JOIN_224/val/all', 'val', native_scores, native_labels, place='native')

model_script_lines = [
    "normalization1 = normalization([128.68140225, 117.6060306, 109.43819699999999 ], [46.829990099999996, 48.069616499999995,  50.7004974])\n",
    "model_optimization_flavor(optimization_level=2, compression_level=2, batch_size=1)\n"
]

# Load the model script to ClientRunner so it will be considered on optimization
runner.load_model_script("".join(model_script_lines))
runner.optimize_full_precision()

# -------------
# run optimized model inference on emulator
# -------------

with runner.infer_context(InferenceContext.SDK_FP_OPTIMIZED) as ctx:
    modified_res = runner.infer(ctx, image_dataset[:, :, :, :])

modified_scores, modified_labels = postproc(modified_res)
aaa = visualize_results(image_dataset[:, :, :, :], './yolo_m2_class_square_JOIN_224/val/all', 'val', modified_scores, modified_labels, place='optimized')

# -------------
# calibrate model using train set
# -------------

images_path_c = './yolo_m2_class_square_JOIN_224/train/all'
images_list = [img_name for img_name in os.listdir(images_path_c) if os.path.splitext(img_name)[1] == ".jpg"]


image_dataset_cal = np.zeros((len(images_list), 224, 224, 3))
for idx, img_name in enumerate(sorted(images_list)):
    img = np.array(Image.open(os.path.join(images_path_c, img_name)))
    img_preproc = preproc(img)
    image_dataset_cal[idx, :, :, :] = img_preproc.numpy()

calib_dataset = image_dataset_cal
hn_layers = runner.get_hn_dict()["layers"]


aaakkkk = [layer for layer in hn_layers if hn_layers[layer]["type"] == "input_layer"]

parts = aaakkkk[0].split("/")
calib_dataset_dict = {f"{parts[0]}/input_layer1": calib_dataset}  # In our case there is only one input layer
runner.optimize(calib_dataset_dict)

# -------------
# run quantized model inference on emulator
# -------------

with runner.infer_context(InferenceContext.SDK_QUANTIZED) as ctx:
    quantized_res = runner.infer(ctx, image_dataset[:, :, :, :])

quantized_scores, quantized_labels = postproc(quantized_res)


csv_path = './total.csv'
file_exists = os.path.isfile(csv_path)
with open(csv_path, 'a', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    # Write a header row (optional)
    if not file_exists:
        csv_writer.writerow(['run_id', 'acc', 'auroc', 'aupr_success', 'aupr', 'fpr', 'tpr', 'aurc', 'eaurc', 'augrc'])
    aaa = visualize_results(image_dataset[:, :, :, :], './yolo_m2_class_square_JOIN_224/val/all', 'val', quantized_scores, quantized_labels, place='quantized')
    akadu = [args.run_id]
    for ii in aaa:
        akadu.append(ii)
    csv_writer.writerow(akadu)

# -------------
# save quantized model and draw wconfidence plot based on per sample csv log
# -------------


quantized_model_har_path = os.path.join(quantized_har_dir, f"{model_name}_quantized_model.har")
runner.save_har(quantized_model_har_path)

for split in ['val']:
    for place in ['native', 'optimized', 'quantized']:
        expected_cols = {'Confidence (%)', 'correct'}
        df = pd.read_csv(f'./per_sample_{place}_{split}.csv')
        if not expected_cols.issubset(df.columns):
            print(f"CSV file {csv_file} does not have the expected columns.")
        
        plt.figure(figsize=(8, 6))
        
        plt.hist(df[df['correct']]['Confidence (%)'], bins=50, color='green',
                 alpha=0.6, density=True, label='Success')
        plt.hist(df[~df['correct']]['Confidence (%)'], bins=50, color='red',
                 alpha=0.6, density=True, label='Error')
        plt.xlim([0, 1])
        plt.xlabel("Confidence")
        plt.ylabel("Density")
        plt.title("Confidence Distribution - {csv_file}")
        plt.legend()
        plt.savefig(f'fmfp_{place}_{split}_plot_{args.run_id}.png')
        plt.close()