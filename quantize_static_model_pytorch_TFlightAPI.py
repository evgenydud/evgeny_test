from onnx_tf.backend import prepare
import tensorflow as tf
import pathlib
import onnx



# first define model class
#loading model
root = '/usr/src/Signals/Sat_proj/yolov5/'
model_path = root+ 'epoch2275_best_SGD_square.pt'
onnx_model1 = model_path.replace('pt', 'onnx')

"""
At this point convert manualy via CLI:
python3.8  export.py --weights 'your weights path'  --batch-size 10 --imgsz 1101 --include onnx --simplify

with training on squared image results on working model
# now you have onnx model in the weight directory
"""

onnx_model = onnx.load(onnx_model1)
#conversion to tflite model (tflite model needed to quantization)
tf_rep = prepare(onnx_model)

tf_rep.export_graph('/'.join(model_path.split('/')[:-1])+'/tf_model_2275')
tf_model = model_path.replace('pt', 'pb')
# quantize tf model
converter = tf.lite.TFLiteConverter.from_saved_model(root+'tf_model_2275/')
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_quant_model = converter.convert()
# save quantized model
tflite_models_dir = pathlib.Path(root)
tflite_model_quant_file = tflite_models_dir/'epoch2275_best_SGD_square.tflite'
tflite_model_quant_file.write_bytes(tflite_quant_model)

#back path to pytorch model

tflite_path = root+ 'epoch2275_best_SGD_square.tflite'

onnx_quantized_export_model_path = root+ 'epoch2275_best_SGD_square_quantized.onnx'
"""
using script command it did not work and using cli it does work
python3.8 -m tf2onnx.convert --tflite 'tflite_path --output onnx_quantized_export_model_path --opset 13
now you have quantized model back in form of onnx
"""
