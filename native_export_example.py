from ppq import *
from ppq.api import *
import os

# modify configuration below:
WORKING_DIRECTORY = 'working'                             # choose your working directory
INPUT_LAYOUT          = 'chw'                             # input data layout, chw or hwc
NETWORK_INPUTSHAPE    = [1, 3, 224, 224]                  # input shape of your network
CALIBRATION_BATCHSIZE = 16                                # batchsize of calibration dataset
EXECUTING_DEVICE      = 'cuda'                            # 'cuda' or 'cpu'.

dataloader = load_calibration_dataset(
    directory    = WORKING_DIRECTORY,
    input_shape  = NETWORK_INPUTSHAPE,
    batchsize    = CALIBRATION_BATCHSIZE,
    input_format = INPUT_LAYOUT)

graph = load_native_graph(import_file = os.path.join(WORKING_DIRECTORY, 'quantized.pkl'))

executor = TorchExecutor(graph=graph, device=EXECUTING_DEVICE)

# 使用executor对模型(量化的或者不量化由上方do_quantize=True控制)进行推理和后处理以验证
output = executor.forward(dataloader[0][0].unsqueeze(0))

print('网络量化结束，正在生成目标文件:')
export_ppq_graph(
        graph=graph, platform=TargetPlatform.PPL_CUDA_INT8,
        graph_save_to = os.path.join(WORKING_DIRECTORY, 'quantized.onnx'),
        config_save_to = os.path.join(WORKING_DIRECTORY, 'quant_cfg.json'))