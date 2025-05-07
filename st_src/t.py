import onnx
print(onnx.helper.printable_graph(onnx.load("../models/model_q.onnx").graph)[:2000])
import onnx, numpy as np
from onnx import numpy_helper

m = onnx.load("../models/model_q.onnx")
for init in m.graph.initializer:
    if init.name.endswith(("weight","bias")) and init.dims and init.dims[-1] in (1,2):
        arr = numpy_helper.to_array(init)
        print(init.name, arr.shape, "→", arr if arr.size<=10 else np.unique(arr)[:5])
