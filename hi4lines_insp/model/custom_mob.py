# model.py
import onnx
import torch
import torch.nn as nn
from onnx2pytorch import ConvertModel

class ConvertedModel(nn.Module):
    """
    Wraps an ONNX graph as a PyTorch nn.Module.
    Loads the ONNX file at init, builds the internal sub-module,
    and delegates forward passes to it.
    """
    def __init__(self, onnx_path: str = "model.onnx"):
        super(ConvertedModel, self).__init__()
        # load the ONNX graph
        onnx_model = onnx.load(onnx_path)
        # convert to an nn.Module
        self.model = ConvertModel(onnx_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

def build_model(onnx_path: str = "model_q.onnx") -> ConvertedModel:
    """
    Factory function: returns a fresh ConvertedModel.
    """
    return ConvertedModel(onnx_path)

if __name__ == "__main__":
    # Quick sanity check: print architecture
    m = build_model()
    print(m)
