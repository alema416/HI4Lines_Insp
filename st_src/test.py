import onnxruntime as ort
import numpy as np
from collections import OrderedDict
from torchvision.models import mobilenet_v2
import torch, onnxruntime as ort
import custom_data 

def load_checkpoint1(pth_path: str):
    # 1) instantiate the exact torchvision MobileNetV2
    model = mobilenet_v2(num_classes=2) #mobilenet(num_classes=2)
    model.eval()

    # 2) load your checkpoint (it may be a dict with 'state_dict' inside)
    ckpt = torch.load(pth_path, map_location="cpu")
    if "state_dict" in ckpt:
        sd = ckpt["state_dict"]
    else:
        sd = ckpt

    # 3) strip off any "module." prefixes and unwanted keys
    clean = OrderedDict()
    for k, v in sd.items():
        # remove DataParallel prefix
        nk = k.replace("module.", "")
        # drop any extra keys your training pipeline may have added
        if nk == "n_averaged":
            continue
        clean[nk] = v

    # 4) load into the model
    model.load_state_dict(clean)
    return model


pt = load_checkpoint1("../models/ckpt.pth").eval()
sess = ort.InferenceSession("../models/model_q.onnx")
_, _, val_loader = custom_data.get_loader_local('/home/alema416/dev/work/HI4Lines_Insp/data/processed/IDID_cropped_224', batch_size=1, input_size=224)

pt_corr = onnx_corr = total = 0
for x,y,_,_ in val_loader:
    total    += y.size(0)
    pt_corr   += (pt(x).argmax(1).cpu().numpy() == y.numpy()).sum()
    onnx_out  = sess.run(None, {"input": x.numpy()})[0]
    onnx_corr += (onnx_out.argmax(1) == y.numpy()).sum()

print("PyTorch test Acc:", pt_corr/total)
print("ONNX    test Acc:", onnx_corr/total)
