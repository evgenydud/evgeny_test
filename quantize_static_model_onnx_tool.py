import onnx
from onnxruntime.quantization import quantize_static, QuantType

# first define model class
#loading model
root = '/Data/Signals/Sat_proj/yolov5/'
model_name = 'epoch2275_best_SGD_square_1.onnx'
model_in = root+ model_name
model_quant = root+'epoch2275_onnx_quant.onnx'
quantized_model = quantize_static(model_in, model_quant,calibration_data_reader = 'CalibrationDataReader')
