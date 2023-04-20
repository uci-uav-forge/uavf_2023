import tensorrt as trt
import numpy as np
from trt_utils import build_engine, allocate_buffers
import pycuda.driver as cuda
import pycuda.autoinit


letterdir = 'letter_detection/letter_model_gpu.onnx'
shapedir = 'yolo/trained_models/seg-v8n.onnx'
TRT_LOGGER = trt.Logger(trt.Logger.INFO)
model1_engine = build_engine(letterdir, TRT_LOGGER)
model2_engine = build_engine(shapedir, TRT_LOGGER)

if model1_engine == None or model2_engine == None:
    print(f"Engines returned are {type(model1_engine)} and {type(model2_engine)}")
h_input1, d_input1, h_output1, d_output1 = allocate_buffers(model1_engine)
h_input2, d_input2, h_output2, d_output2 = allocate_buffers(model2_engine)

model1_context = model1_engine.create_execution_context()
model2_context = model2_engine.create_execution_context()

