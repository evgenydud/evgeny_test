testing_dataset = torchvision.datasets.CIFAR10(root='./data',
                                                        train=False,
                                                        download=True,
                                                        transform=transforms.Compose([
                                                            transforms.ToTensor(),
                                                            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                                                                 (0.2023, 0.1994, 0.2010))
                                                        ]))

testing_dataloader = torch.utils.data.DataLoader(testing_dataset,
                                                      batch_size=1,
                                                      shuffle=False,
                                                      num_workers=1)
calibrator = torch_tensorrt.ptq.DataLoaderCalibrator(testing_dataloader,
                                              cache_file='./calibration.cache',
                                              use_cache=False,
                                              algo_type=torch_tensorrt.ptq.CalibrationAlgo.ENTROPY_CALIBRATION_2,
                                              device=torch.device('cuda:0'))

compile_spec = {
         "inputs": [torch_tensorrt.Input((1, 3, 32, 32))],
         "enabled_precisions": {torch.float, torch.half, torch.int8},
         "calibrator": calibrator,
         "device": {
             "device_type": torch_tensorrt.DeviceType.GPU,
             "gpu_id": 0,
             "dla_core": 0,
             "allow_gpu_fallback": False,
             "disable_tf32": False
         }
     }
trt_mod = torch_tensorrt.compile(model, compile_spec)