import engine as eng
import Inference as inf
import keras
import numpy as np
from PIL import Image
import tensorrt as trt 


input_file_path = ‘munster_000172_000019_leftImg8bit.png’
onnx_file = "semantic.onnx"
serialized_plan_fp32 = "semantic.plan"
HEIGHT = 512
WIDTH = 1024

image = np.asarray(Image.open(input_file_path))


engine = eng.load_engine(trt_runtime, serialized_plan_fp32)
h_input, d_input, h_output, d_output, stream = inf.allocate_buffers(engine, 1, trt.float32)
out = inf.do_inference(engine, im, h_input, d_input, h_output, d_output, stream, 1, HEIGHT, WIDTH)
out = color_map(out)

colorImage_trt = Image.fromarray(out.astype(np.uint8))
colorImage_trt.save(“trt_output.png”)

semantic_model = keras.models.load_model('/path/to/semantic_segmentation.hdf5')
out_keras= semantic_model.predict(im.reshape(-1, 3, HEIGHT, WIDTH))

out_keras = color_map(out_keras)
colorImage_k = Image.fromarray(out_keras.astype(np.uint8))
colorImage_k.save(“keras_output.png”)