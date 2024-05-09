import torch
import numpy
from torch.quantization import quantize_fx

# Static quantization of a model consists of the following steps:

#     Fuse modules
#     Insert Quant/DeQuant Stubs
#     Prepare the fused module (insert observers before and after layers)
#     Calibrate the prepared module (pass it representative data)
#     Convert the calibrated module (replace with quantized version)
m_path = '/Data/Signals/Sat_proj/yolov5/epoch2275_best_SGD_square.pt'


backend = "fbgemm"  # running on a x86 CPU. Use "qnnpack" if running on ARM.
# should I load yolo weights?
m = torch.hub.load('ultralytics/yolov5', 'custom', path=m_path)

## FX GRAPH
m.eval()
qconfig_dict = {"": torch.quantization.get_default_qconfig(backend)}
# Prepare
model_prepared = quantize_fx.prepare_fx(m, qconfig_dict)
# Calibrate - Use representative (validation) data.!!!
with torch.inference_mode():
    for _ in range(10):
        x = torch.rand(1, 2, 28, 28)
        model_prepared(x)
# quantize
model_quantized = quantize_fx.convert_fx(model_prepared)
