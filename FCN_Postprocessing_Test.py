# -*- coding: utf-8 -*-
"""
Created on Wed Nov 10 16:34:57 2021

@author: ander

FCN_ResNet50 Test
"""

##  Importar Librerias
import numpy as np
# import glob
# import os
# import onnx
import onnxruntime as ort
import time
import numpy
import cv2 

# from onnx import numpy_helper
from PIL import Image

import torchvision.transforms as transforms
to_tensor = transforms.ToTensor()

# import torch

# from matplotlib.pyplot import imshow

from FCN_postprocess_func import *

##  Definiciones
preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

## Cargar y preprocesar imagen
img = Image.open("images/FCN/000000017968.jpg")
orig_tensor = np.asarray(img)
orig_tensor = cv2.cvtColor(orig_tensor, cv2.COLOR_BGR2RGB)
img_data = preprocess(img)
img_data = img_data.unsqueeze(0)
img_data = img_data.detach().cpu().numpy()

## Visualizar imagen Cargada
cv2.imshow('Original Image',orig_tensor)
cv2.waitKey(0)
cv2.destroyAllWindows()

##  Cargar Modelo
Model_path="Modelos/FCN_ResNet50/fcn-resnet50-11.onnx"

# ort_session = ort.InferenceSession(Model_path,
#                                     providers=["CUDAExecutionProvider"])

ort_session = ort.InferenceSession(Model_path,
                                    providers=["CPUExecutionProvider"])

##  Inferencia

inference_start=time.time()
ort_inputs={ort_session.get_inputs()[0].name: (img_data)}#,
                # ort_session.get_inputs()[1].name: image_size}
ort_outs = ort_session.run(None, ort_inputs)

output, aux = ort_outs

print("Output shape:", list(map(lambda detection: detection.shape, ort_outs)))
  
inference_time=time.time()-inference_start
    
print("Tiempo de inferencia {:.4f} ms".format(inference_time*1000))

##  Postprocesar Salida
conf, result_img, blended_img, raw_labels = visualize_output(orig_tensor, output[0])

##  Mostrar mascara
cv2.imshow('Mascara',result_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

##  Mostrar superposicion
cv2.imshow('Superposicion',blended_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
