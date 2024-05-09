import torch
import pickle
from PIL import Image
import time
import scipy

x = torch.randn(3, 2)
y = torch.view_as_complex(x)
torch.save(y, 'complex_tensor.pt')
x = torch.load('complex_tensor.pt')
print('y')

img_path = '/Data/Signals/Sat_proj/data/20_04_2022/images/test/'
img = Image.open(img_path+'17513_22000000mb.png')
start = time.time()
model = torch.hub.load('.', 'custom', path='epoch2275_best_SGD_square.pt', source='local')
results = model(img)
print('\nInference time in ms: ', '%.1f' % (1000*(time.time()-start)))
print(results.xywhn[0][:,:6])  # im predictions (tensor)

