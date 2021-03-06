#https://docs.nvidia.com/deeplearning/tensorrt/api/python_api/infer/Core/Builder.html#tensorrt.Builder

#https://medium.com/@fanzongshaoxing/accelerate-pytorch-model-with-tensorrt-via-onnx-d5b5164b369

import tensorrt as trt

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
trt_runtime = trt.Runtime(TRT_LOGGER)


def build_engine(onnx_path, shape=[1, 224, 224, 3]):
    """
    This is the function to create the TensorRT engine
    Args:
        onnx_path : Path to onnx_file.
        shape : Shape of the input of the ONNX file.
    """
    with trt.Builder(TRT_LOGGER) as builder, \
            builder.create_network(1) as network, \
            builder.create_builder_config() as config, \
            trt.OnnxParser(network, TRT_LOGGER) as parser:
        config.max_workspace_size = 512 << 20
        builder.max_batch_size = 1
 #Flags para la optimizacion
        config.set_flag(trt.BuilderFlag.FP16) #Optimizacion de tamano
        config.set_flag(trt.BuilderFlag.SPARSE_WEIGHTS)#Allow the builder to examine weights and use optimized functions when weights have suitable sparsity

        #config.set_flag(trt.BuilderFlag.DISABLE_TIMING_CACHE)
        with open(onnx_path, 'rb') as model:
            parser.parse(model.read())
        network.get_input(0).shape = shape
        engine = builder.build_engine(network, config)
        return engine


def save_engine(engine, file_name):
    buf = engine.serialize()
    with open(file_name, 'wb') as f:
        f.write(buf)


def load_engine(trt_runtime, plan_path):
    with open(plan_path, 'rb') as f:
        engine_data = f.read()
    engine = trt_runtime.deserialize_cuda_engine(engine_data)
    return engine
