# -*- coding: utf-8 -*-
"""
Created on Mon Nov 15 11:25:17 2021

@author: ander
https://github.com/onnx/models/blob/master/vision/object_detection_segmentation/yolov4/dependencies/inference.ipynb
"""
##  Importar Librerias
import numpy as np
# from PIL import Image
import time

import cv2

# import onnx
import onnxruntime as ort
# from onnx import numpy_helper


## Generar Imagenes
Imagen=np.zeros((1,3,1024,1024),dtype=np.float32)
Patron=np.zeros((1,3,224,224),dtype=np.float32)

##  Inferencia

#Model_path="Modelos\Comar Models\CoMAr_CNN_V02.onnx"
Model_path="Modelos\Comar Models\Vgg16byMatlab.onnx"

#sess = ort.InferenceSession(Model_path,providers=["CPUExecutionProvider"])
sess = ort.InferenceSession(Model_path,providers=["CUDAExecutionProvider"])

outputs = sess.get_outputs()
output_names = list(map(lambda output: output.name, outputs))
input_name0 = sess.get_inputs()[0].name
#input_name1 = sess.get_inputs()[1].name

inference_start=time.time()
#detections = sess.run(output_names, {input_name0: Imagen, input_name1: Patron})
detections = sess.run(output_names, {input_name0: Patron})
inference_time=time.time()-inference_start

print(f'TIempo de inferencia: {inference_time} s')