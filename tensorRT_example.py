from pytorch_quantization import tensor_quant
import torch
import torchvision
import warnings
warnings.filterwarnings('ignore')

# Generate random input. With fixed seed 12345, x should be
# tensor([0.9817, 0.8796, 0.9921, 0.4611, 0.0832, 0.1784, 0.3674, 0.5676, 0.3376, 0.2119])
torch.manual_seed(12345)
x = torch.rand(10)

# fake quantize tensor x. fake_quant_x will be
# tensor([0.9843, 0.8828, 0.9921, 0.4609, 0.0859, 0.1797, 0.3672, 0.5703, 0.3359, 0.2109])
fake_quant_x = tensor_quant.fake_tensor_quant(x, x.abs().max())

# quantize tensor x. quant_x will be
# tensor([126., 113., 127.,  59.,  11.,  23.,  47.,  73.,  43.,  27.])
# with scale=128.0057
quant_x, scale = tensor_quant.tensor_quant(x, x.abs().max())

from pytorch_quantization.tensor_quant import QuantDescriptor
from pytorch_quantization.nn.modules.tensor_quantizer import TensorQuantizer

quant_desc = QuantDescriptor(num_bits=4, fake_quant=False, axis=(0), unsigned=True)
quantizer = TensorQuantizer(quant_desc)

torch.manual_seed(12345)
x = torch.rand(10, 3, 64, 64)

quant_x = quantizer(x)


from pytorch_quantization import quant_modules
model = torchvision.models.resnet50()
for name, module in model.named_modules():
    if name.endswith('_input_quantizer'):
        module.enable_calib()
        module.disable_quant()  # Use full precision data to calibrate

# Feeding data samples
model(x)
# ...

# Finalize calibration
for name, module in model.named_modules():
    if name.endswith('_input_quantizer'):
        module.load_calib_amax()
        module.enable_quant()

# If running on GPU, it needs to call .cuda() again because new tensors will be created by calibration process
model.cuda()
print('\nAll went well\n')
