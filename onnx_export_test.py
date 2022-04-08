# -*- coding: utf-8 -*-
"""
Created on Wed Nov 10 16:34:57 2021

@author: ander

Onnx Export Test
"""

##  Importar Librerias
import numpy as np
import glob
import os
import onnx
import onnxruntime as ort
import time
import numpy

from onnx import numpy_helper
from PIL import Image

import torchvision.transforms as transforms
to_tensor = transforms.ToTensor()

import torch

##  Definiciones
preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


## Dummy image
Patron=np.zeros((1,3,224,224),dtype=np.float32)
Imagen=np.zeros((1,3,1024,1024),dtype=np.float32)
##  Cargar Modelo
Model_path="Modelos\Comar Models\CoMAr_CNN_V02.onnx"
#Model_path=r"Modelos\resnet18v2\resnet18_v1_7.onnx"
#Model_path=r"Modelos\Comar Models\resnet18_by_Matlab.onnx"

##  definir entorno de inferencia   

ort_session = ort.InferenceSession(Model_path,
                                    providers=["CUDAExecutionProvider"],
                                    provider_options=[{"device_id": 0}])

# ort_session = ort.InferenceSession(Model_path,
#                                     providers=["CPUExecutionProvider"])

outputs = ort_session.get_outputs()
output_names = list(map(lambda output: output.name, outputs))
input_name0 = ort_session.get_inputs()[0].name
input_name1 = ort_session.get_inputs()[1].name


##  Inferencia

n_sim=500
time_array=numpy.zeros(n_sim)
for i in range (n_sim): 

    inference_start=time.time()
    ort_inputs={ort_session.get_inputs()[0].name: (Patron)}#,
                # ort_session.get_inputs()[1].name: image_size}
    #ort_outs = ort_session.run(None, ort_inputs)
    detections = ort_session.run(output_names, {input_name0: Imagen, input_name1: Patron})
    
    inference_time=time.time()-inference_start
    
    print("Tiempo de inferencia {:.4f} ms".format(inference_time*1000))
    time_array[i]=inference_time
    
worst_time=numpy.amax(time_array[1:n_sim])  
mean_time=numpy.mean(time_array[1:n_sim])       
print("Tiempo medio {:.4f} ms   Peor Tiempo= {:.4f} ms".format(mean_time*1000,
                                                               worst_time*1000)) 
