import torch
import os
from vitsolv import ViTVNet, get_3DReg_config
from usolv import ResidualUNetSE3D
import sys

"""
Script for converting the Pytorch models to Onnx.

Usage:

python export_to_onnx.py path/to/pytorch/model unet
python export_to_onnx.py path/to/pytorch/model vitvnet
"""

model_path = os.path.join(os.getcwd(), sys.argv[1])
model_type = sys.argv[2]

if model_type == "vitvnet":
    config = get_3DReg_config()
    model = ViTVNet(config=config)
    model = torch.nn.DataParallel(model)
    model.cuda()
elif model_type == "unet":
    model = ResidualUNetSE3D(
            in_channels=1,
            f_maps=[16, 32, 64, 128, 256]

            )
    model = torch.nn.DataParallel(model)
    model.cuda()
else:
    raise NotImplementedError


model.load_state_dict(torch.load(model_path))
model.eval()

dummy_input = torch.randn(1, 1, 256, 256, 256).cuda()  

onnx_model_path = os.path.join(os.getcwd(), "model.onnx")
torch.onnx.export(
    model.module,        
    dummy_input,          
    onnx_model_path,       
    export_params=True,     
    opset_version=11,        
    input_names=["input"],    
    output_names=["output"],   
    dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}}
                                        )

