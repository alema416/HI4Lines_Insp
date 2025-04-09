import degirum as dg
import degirum_tools
from pprint import pprint
import detection_eval

model_name = 'yolov11n_vanilla_simplified'

# Load the model from the model zoo.
# Replace '<path_to_model_zoo>' with the directory containing your model assets.

model = dg.load_model(
    model_name=model_name,
    inference_host_address='@local',
    zoo_url="/home/amax/GitHub/hailo_examples/models/"
)

evaluator = detection_eval.ObjectDetectionModelEvaluator(
    model,
    show_progress=False,
    classmap={0: 0, 1: 1},  
    pred_path='./b.json' 
)

results_train = evaluator.evaluate('./assets/detection_train_set/images/', './assets/annotations_train.json', 0)
results_val = evaluator.evaluate('./assets/detection_val_set/images/', './assets/annotations_val_multiclass_True.json', 0)
results_test = evaluator.evaluate('./assets/detection_test_set/images/', './assets/annotations_test_multiclass_True.json', 0)

speed = degirum_tools.inference_support.model_time_profile(model)
print(speed)
