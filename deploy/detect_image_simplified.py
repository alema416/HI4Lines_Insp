import degirum as dg
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
inference_result = model('./assets/scene.jpg')

# Pretty print the detection results.
pprint(inference_result.results)


# Display the image with overlayed detection results.
cv2.imshow("AI Inference", inference_result.image_overlay)

# Wait for the user to press 'x' or 'q' to exit.
while True:
    key = cv2.waitKey(0) & 0xFF  # Wait indefinitely until a key is pressed.
    if key == ord('x') or key == ord('q'):
        break
cv2.destroyAllWindows()  # Close all OpenCV windows.
