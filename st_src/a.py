import onnx
from onnx2pytorch import ConvertModel

# Load the specific ONNX model and convert it to a PyTorch module
onnx_model = onnx.load("../models/model_q.onnx")
model = ConvertModel(onnx_model)

if __name__ == "__main__":
    # Quick sanity check: print the converted architecture
    print(model)
