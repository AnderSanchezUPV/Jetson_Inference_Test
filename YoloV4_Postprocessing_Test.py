# -*- coding: utf-8 -*-
"""
Created on Mon Nov 15 11:25:17 2021

@author: ander
https://github.com/onnx/models/blob/master/vision/object_detection_segmentation/yolov4/dependencies/inference.ipynb
"""
## Limpiar el workspace y la consola
from IPython import get_ipython
get_ipython().magic('reset -sf')
get_ipython().magic('clear')

##  Importar Librerias
import numpy as np
# from PIL import Image
import time

import cv2

# import onnx
import onnxruntime as ort
# from onnx import numpy_helper

from YoloV4_postprocess_func import *

# import os

# from matplotlib.pyplot import imshow
# from matplotlib import pyplot as plt

# from scipy import special
# import colorsys
# import random


##  Definicion de funciones
def image_preprocess(image, target_size, gt_boxes=None):

    ih, iw = target_size
    h, w, _ = image.shape

    scale = min(iw/w, ih/h)
    nw, nh = int(scale * w), int(scale * h)
    image_resized = cv2.resize(image, (nw, nh))

    image_padded = np.full(shape=[ih, iw, 3], fill_value=128.0)
    dw, dh = (iw - nw) // 2, (ih-nh) // 2
    image_padded[dh:nh+dh, dw:nw+dw, :] = image_resized
    image_padded = image_padded / 255.
    return image_padded




##  Tomar Imagen
    #Desde archivo
# original_image = cv2.imread("images/Yolo/kite.jpg")
# original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

    #Desde camara
cam = cv2.VideoCapture(0)
cv_flag ,original_image = cam.read() 
   
##  Preprocesar Imagen
input_size = 416

original_image_size = original_image.shape[:2]

image_data = image_preprocess(np.copy(original_image), [input_size, input_size])
image_data = image_data[np.newaxis, ...].astype(np.float32)

print("Preprocessed image shape:",image_data.shape) # shape of the preprocessed inp

cv2.imshow('Original Image',original_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

##  Inferencia

Model_path="Modelos/YoloV4/yolov4.onnx"
sess = ort.InferenceSession(Model_path,
                            providers=["CUDAExecutionProvider"])
model_start=time.time()
outputs = sess.get_outputs()
output_names = list(map(lambda output: output.name, outputs))
input_name = sess.get_inputs()[0].name

detections = sess.run(output_names, {input_name: image_data})
print("Output shape:", list(map(lambda detection: detection.shape, detections)))

##  PostProcesar Imagen

#   Las funciones de postprocesado se importan desde: YoloV4_postprocess_func
ANCHORS = "Modelos/YoloV4/yoloV4_anchors.txt"
STRIDES = [8, 16, 32]
XYSCALE = [1.2, 1.1, 1.05]

ANCHORS = get_anchors(ANCHORS)
STRIDES = np.array(STRIDES)

pred_bbox = postprocess_bbbox(detections, ANCHORS, STRIDES, XYSCALE)
bboxes = postprocess_boxes(pred_bbox, original_image_size, input_size, 0.25)
bboxes = nms(bboxes, 0.213, method='nms')
image = draw_bbox(original_image, bboxes)

model_end=time.time()-model_start
print("Tiempo de inferencia {:.4f} ms".format(model_end*1000))
##  Mostrar Resultado
cv2.imshow('YoloV4 Output',image)
cv2.waitKey(0)
cam.release()
cv2.destroyAllWindows()