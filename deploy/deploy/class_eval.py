import degirum as dg
import degirum_tools
from pprint import pprint
import class_eval_sec
import argparse

#default='./output_lp_earlier_flat/', default=4, 
parser = argparse.ArgumentParser(description='Rethinking CC for FP')

parser.add_argument('--model', default='resnet_v1_18_custom_code', type=str, help='model name')
args = parser.parse_args()

model_name = args.model

model = dg.load_model(
    model_name=model_name,
    inference_host_address='@local',
    zoo_url="/home/amax/GitHub/hailo_examples/models/"
)

evaluator_tr = class_eval_sec.ImageClassificationModelEvaluator(
    model,
    show_progress=False, split='train',
    top_k=[1],  # Evaluate top-1 and top-5 accuracy
    foldermap={0: "broken", 1: "healthy"}  # Mapping class IDs to folders
)
evaluator_v = class_eval_sec.ImageClassificationModelEvaluator(
    model,
    show_progress=False, split='val',
    top_k=[1],  # Evaluate top-1 and top-5 accuracy
    foldermap={0: "broken", 1: "healthy"}  # Mapping class IDs to folders
)
evaluator_te = class_eval_sec.ImageClassificationModelEvaluator(
    model,
    show_progress=False, split='test',
    top_k=[1],  # Evaluate top-1 and top-5 accuracy
    foldermap={0: "broken", 1: "healthy"}  # Mapping class IDs to folders
)

print(f'WORKING ON: train set')
results_train = evaluator_tr.evaluate('./assets/classification_train_set/', None, -1)
print(f'WORKING ON: val set')
results_eval = evaluator_v.evaluate('./assets/classification_eval_set/', None, -1)
print(f'WORKING ON: test set')
results_test = evaluator_te.evaluate('./assets/classification_test_set/', None, -1)

print(f'train_set top1 acc: {results_train[0][0]:.5f}%')
print(f'validation_set top1 acc: {results_eval[0][0]:.5f}%')
print(f'test_set top1 acc: {results_test[0][0]:.5f}%')

print(f'train_set per_class accuracies: {results_train[1][0][0]:.5f}%, {results_train[1][1][0]:.5f}%')
print(f'validation_set per_class accuracies: {results_eval[1][0][0]:.5f}%, {results_eval[1][1][0]:.5f}%')
print(f'test_set per_class accuracies: {results_test[1][0][0]:.5f}%, {results_test[1][1][0]:.5f}%')

print(f'SPECIAL_PRINTacctrain {results_train[0][0]:.3f}')
print(f'SPECIAL_PRINTaccval {results_eval[0][0]:.3f}')
print(f'SPECIAL_PRINTacctest {results_test[0][0]:.3f}')