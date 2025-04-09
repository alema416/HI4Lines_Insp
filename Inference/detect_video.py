import degirum as dg
import degirum_tools
from pprint import pprint
import cv2

model_name = 'yolov11n_vanilla_simplified'

# Load the model from the model zoo.
# Replace '<path_to_model_zoo>' with the directory containing your model assets.

model = dg.load_model(
    model_name=model_name,
    inference_host_address='@local',
    zoo_url="/home/amax/GitHub/hailo_examples/models/"
)

# Run inference on the input image.
# Replace '<path_to_cat_image>' with the actual path to your cat image.
#inference_result = model('./assets/scene.jpg')
inference_results = degirum_tools.predict_stream(model, './assets/output.mp4')
#print(degirum_tools.get_video_stream_properties(inference_results))
# Pretty print the detection results.

#pprint(inference_result.results)
'''

# Display the image with overlayed detection results.
cv2.imshow("AI Inference", inference_results.image_overlay)

# Wait for the user to press 'x' or 'q' to exit.
while True:
    key = cv2.waitKey(0) & 0xFF  # Wait indefinitely until a key is pressed.
    if key == ord('x') or key == ord('q'):
        break
cv2.destroyAllWindows()  # Close all OpenCV windows.
'''
# display inference results
# Press 'x' or 'q' to stop
with degirum_tools.Display("AI Camera") as display:
    for inference_result in inference_results:
        display.show(inference_result)
