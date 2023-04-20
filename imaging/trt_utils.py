import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import onnx
import numpy as np

def build_engine(onnx_file_path, trt_logger):
    with trt.Builder(trt_logger) as builder,  builder.create_builder_config() as config, builder.create_network(1 <<  int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)) as network, trt.OnnxParser(network, trt_logger) as parser:
        model = onnx.load(onnx_file_path)
        profile = builder.create_optimization_profile()
        for i in model.graph.input:
            d = i.type.tensor_type.shape.dim
            d = tuple(x.dim_value for x in d)
            profile.set_shape(i.name,d,d,d)
        config.add_optimization_profile(profile)

        config.max_workspace_size = 1 << 11  # 4096MiB
        builder.max_batch_size = 2
        with open(onnx_file_path, 'rb') as model:
            parser.parse(model.read())
        return builder.build_engine(network, config)

def allocate_buffers(engine):
    h_input = cuda.pagelocked_empty(trt.volume(engine.get_binding_shape(0)), dtype=np.float32)
    h_output = cuda.pagelocked_empty(trt.volume(engine.get_binding_shape(1)), dtype=np.float32)
    d_input = cuda.mem_alloc(h_input.nbytes)
    d_output = cuda.mem_alloc(h_output.nbytes)
    return h_input, d_input, h_output, d_output

def do_inference(engine, context, h_input, d_input, h_output, d_output):
    # Transfer input data to the GPU.
    cuda.memcpy_htod(d_input, h_input)
    # Run inference.
    context.execute(batch_size=1, bindings=[int(d_input), int(d_output)])
    # Transfer predictions back from the GPU.
    cuda.memcpy_dtoh(h_output, d_output)
    return h_output

