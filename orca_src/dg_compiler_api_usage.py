import requests
import json
import argparse
import os

def determine_url(model_file, port=8535):
    """Determine the compiler mode URL based on the model file extension."""
    if model_file.endswith('.pt'):
        return f'http://ml-net:{port}/yolocompile'
    elif model_file.endswith(('.onnx', '.tflite')):
        return f'http://ml-net:{port}/generalcompile'
    else:
        raise ValueError("Unsupported model file format. Supported formats are '.pt', '.onnx', '.tflite'.")


def load_json(json_file):
    """Load input parameters from a JSON file."""
    if not os.path.isfile(json_file):
        raise FileNotFoundError(f"JSON file '{json_file}' not found.")
    with open(json_file, 'r') as f:
        return json.load(f)


def prepare_files(model_file, class_file=None, images_folder=None):
    """Prepare files for the POST request."""
    files = {'checkpoint_file': open(model_file, 'rb')}

    if class_file:
        if not os.path.isfile(class_file):
            raise FileNotFoundError(f"Class file '{class_file}' not found.")
        files['class_names'] = open(class_file, 'rb')

    if images_folder:
        if not os.path.isdir(images_folder):
            raise FileNotFoundError(f"Images folder '{images_folder}' not found.")

        image_files = [
            f for f in sorted(os.listdir(images_folder))
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ]
        for idx, image_file in enumerate(image_files, start=1):
            image_path = os.path.join(images_folder, image_file)
            files[f'image{idx}'] = open(image_path, 'rb')

    return files


def send_request(url, input_params, files):
    """Send the POST request with the given URL, parameters, and files."""
    response = requests.post(url, files=files, data={'input_params': json.dumps(input_params)})

    # Close all opened files
    for file in files.values():
        file.close()

    # Handle response
    print(f'Status Code: {response.status_code}')
    try:
        print(f'Response JSON: {response.json()}')
    except json.JSONDecodeError:
        print('Response content is not in JSON format.')


def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Send a POST request with model and parameters.')
    parser.add_argument('--json_file', required=True, help='Path to the JSON file containing input parameters.')
    parser.add_argument('--model_file', required=True, help='Path to the model checkpoint file (e.g., .pt, .onnx, .tflite).')
    parser.add_argument('--class_file', help='Optional path to the class names file (e.g., .yaml file).')
    parser.add_argument('--calib_images_folder', help='Optional path to a folder of images.')
    parser.add_argument('--port', type=int, default=8535, help='Port number of the compiler server.')
    args = parser.parse_args()

    try:
        # Determine the URL based on the model file
        url = determine_url(args.model_file, args.port)

        # Load input parameters from the JSON file
        input_params = load_json(args.json_file)
        input_params['model_file_name'] = os.path.basename(args.model_file)

        # Prepare files for the POST request
        files = prepare_files(args.model_file, args.class_file, args.calib_images_folder)

        # Send the request
        send_request(url, input_params, files)

    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}")


if __name__ == '__main__':
    main()
