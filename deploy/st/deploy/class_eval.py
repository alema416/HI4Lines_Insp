import degirum as dg
import degirum_tools
from pprint import pprint
import class_eval_src
import argparse
from hydra import initialize, compose

with initialize(config_path="../../configs/"):
    cfg = compose(config_name="hw_classifier")  # exp1.yaml with defaults key

model_name = cfg.classifier.modelname

model = dg.load_model(
    model_name=model_name,
    inference_host_address='@local',
    zoo_url= cfg.classifier.model_zoo_dir
)

evaluator_tr = class_eval_src.ImageClassificationModelEvaluator(
    model,
    show_progress=False, split='train',
    top_k=[1],  # Evaluate top-1 and top-5 accuracy
    foldermap={0: "broken", 1: "healthy"}  # Mapping class IDs to folders
)
evaluator_v = class_eval_src.ImageClassificationModelEvaluator(
    model,
    show_progress=False, split='val',
    top_k=[1],  # Evaluate top-1 and top-5 accuracy
    foldermap={0: "broken", 1: "healthy"}  # Mapping class IDs to folders
)
evaluator_te = class_eval_src.ImageClassificationModelEvaluator(
    model,
    show_progress=False, split='test',
    top_k=[1],  # Evaluate top-1 and top-5 accuracy
    foldermap={0: "broken", 1: "healthy"}  # Mapping class IDs to folders
)

print(f'WORKING ON: train set')
results_train = evaluator_tr.evaluate(cfg.classifier.train_set_dir, None, -1)
print(f'WORKING ON: val set')
results_eval = evaluator_v.evaluate(cfg.classifier.val_set_dir, None, -1)
print(f'WORKING ON: test set')
results_test = evaluator_te.evaluate(cfg.classifier.test_set_dir, None, -1)

print(f'train_set top1 acc: {results_train[0][0]:.5f}%')
print(f'validation_set top1 acc: {results_eval[0][0]:.5f}%')
print(f'test_set top1 acc: {results_test[0][0]:.5f}%')

print(f'train_set per_class accuracies: {results_train[1][0][0]:.5f}%, {results_train[1][1][0]:.5f}%')
print(f'validation_set per_class accuracies: {results_eval[1][0][0]:.5f}%, {results_eval[1][1][0]:.5f}%')
print(f'test_set per_class accuracies: {results_test[1][0][0]:.5f}%, {results_test[1][1][0]:.5f}%')

print(f'SPECIAL_PRINTacctrain {results_train[0][0]:.3f}')
print(f'SPECIAL_PRINTaccval {results_eval[0][0]:.3f}')
print(f'SPECIAL_PRINTacctest {results_test[0][0]:.3f}')
