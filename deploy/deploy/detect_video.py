import degirum as dg
import degirum_tools
from pprint import pprint
import cv2

model_name = 'yolov11n_vanilla_simplified'

model = dg.load_model(
    model_name=model_name,
    inference_host_address='@local',
    zoo_url="/home/amax/GitHub/hailo_examples/models/"
)

inference_results = degirum_tools.predict_stream(model, './assets/output.mp4')

with degirum_tools.Display("AI Camera") as display:
    for inference_result in inference_results:
        display.show(inference_result)
