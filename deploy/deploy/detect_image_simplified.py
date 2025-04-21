import degirum as dg
from pprint import pprint
import cv2

model_name = 'yolov11n_vanilla_simplified'

model = dg.load_model(
    model_name=model_name,
    inference_host_address='@local',
    zoo_url="/home/amax/GitHub/hailo_examples/models/"
)

inference_result = model('./assets/scene.jpg')


pprint(inference_result.results)


cv2.imshow("AI Inference", inference_result.image_overlay)

while True:
    key = cv2.waitKey(0) & 0xFF  # Wait indefinitely until a key is pressed.
    if key == ord('x') or key == ord('q'):
        break
cv2.destroyAllWindows()  # Close all OpenCV windows.
