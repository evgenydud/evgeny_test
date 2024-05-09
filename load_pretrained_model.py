import torch
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
print('y')
torch.save(model.state_dict(), '/Data/Signals/Sat_proj/yolov5/pretrained.pt')
img = '/Data/Signals/Sat_proj/yolov5/zidan.jpeg'
results = model(img)
results.print()  # or .show(), .save(), .crop(), .pandas(), etc.



