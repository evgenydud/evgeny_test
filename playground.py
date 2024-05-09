import torch
from torch.quantization.observer import MinMaxObserver, MovingAverageMinMaxObserver, HistogramObserver
import matplotlib.pyplot as plt
import numpy as np

C, L = 3, 4
normal = torch.distributions.normal.Normal(0,1)
inputs = [normal.sample((C, L)), normal.sample((C, L))]
print(inputs)
observers = [MinMaxObserver(), MovingAverageMinMaxObserver(), HistogramObserver()]
for obs in observers:
  for x in inputs:
      obs(x)
  print(obs.__class__.__name__, obs.calculate_qparams())

act =  torch.distributions.pareto.Pareto(1, 10).sample((1,1024))
weights = torch.distributions.normal.Normal(0, 0.12).sample((3, 64, 7, 7)).flatten()

def get_symmetric_range(x):
  beta = torch.max(x.max(), x.min().abs())
  return -beta.item(), beta.item()

def get_affine_range(x):
  return x.min().item(), x.max().item()

def plot(plt, data, scheme):
  boundaries = get_affine_range(data) if scheme == 'affine' else get_symmetric_range(data)
  a, _, _ = plt.hist(data, density=True, bins=100)
  ymin, ymax = np.quantile(a[a>0], [0.25, 0.95])
  plt.vlines(x=boundaries, ls='--', colors='purple', ymin=ymin, ymax=ymax)

fig, axs = plt.subplots(2,2)
plot(axs[0, 0], act, 'affine')
axs[0, 0].set_title("Activation, Affine-Quantized")

plot(axs[0, 1], act, 'symmetric')
axs[0, 1].set_title("Activation, Symmetric-Quantized")

plot(axs[1, 0], weights, 'affine')
axs[1, 0].set_title("Weights, Affine-Quantized")

plot(axs[1, 1], weights, 'symmetric')
axs[1, 1].set_title("Weights, Symmetric-Quantized")
plt.show()

for qscheme in [torch.per_tensor_affine, torch.per_tensor_symmetric]:
  obs = MovingAverageMinMaxObserver(qscheme=qscheme)
  for x in inputs: obs(x)
  print(f"\nQscheme: {qscheme} | {obs.calculate_qparams()}")

from torch.quantization.observer import MovingAveragePerChannelMinMaxObserver
obs = MovingAveragePerChannelMinMaxObserver(ch_axis=0)  # calculate qparams for all `C` channels separately
for x in inputs:
    obs(x)
print('obs calculated params: ', obs.calculate_qparams())

backend = 'fbgemm' if 'x86' else 'qnnpack'
qconfig = torch.quantization.get_default_qconfig(backend)
torch.backends.quantized.engine = backend

my_qconfig = torch.quantization.QConfig(activation=MovingAverageMinMaxObserver.with_args(qscheme=torch.per_tensor_affine),
                                        weight=MovingAveragePerChannelMinMaxObserver.with_args(qscheme=torch.qint8))
print('My qconfig: ', my_qconfig)

