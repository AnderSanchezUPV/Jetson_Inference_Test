# -*- coding: utf-8 -*-
"""
Created on Wed Nov 10 16:34:57 2021

@author: ander

Onnx Export Test
"""

## Librerias
import numpy as np
# import onnx
import onnxruntime as ort
import time
import cv2


## Definir Arrays para emular imagenes.

img=cv2.imread(r"images/Screws/0093.tiff",cv2.IMREAD_GRAYSCALE)
img_copy=np.copy(img)
img=np.expand_dims(img,0)
img=np.expand_dims(img,0)
##  Cargar Modelo
#Model_path=r"Modelos\Comar Models\CoMAr_CNN_V02.onnx"
Model_path=r"Modelos/Comar_Models/Comar Models/ligthScrewNetV2.onnx"

##  Definir entorno de inferencia   

#   - Path al modelo
#   - Ejecutar en Grafica (CUDA) o en procesador
#   - Para (CUDA), en caso de varias graficas, seleccionar el dispositivo de jecucion
ort_session = ort.InferenceSession(Model_path, 
                                    providers=["CUDAExecutionProvider"])

# ort_session = ort.InferenceSession(Model_path,
#                                     providers=["CPUExecutionProvider"])

#   Definir los nombres de las variables y entradas de salida para la inferencia. 
#   Se extraen de la informacion del modelo

outputs = ort_session.get_outputs()
output_names = list(map(lambda output: output.name, outputs))
input_name0 = ort_session.get_inputs()[0].name
#input_name1 = ort_session.get_inputs()[1].name

##  Inferencia

#
detections = ort_session.run(output_names, {input_name0: img.astype(np.float32)}) 

drawed_img=cv2.circle(img_copy,(detections[0][0][0],detections[0][0][1]),15,(255,0,0),5) 

cv2.imshow('Screw Test',drawed_img)
k=cv2.waitKey(0)             
cv2.destroyAllWindows()
        ##  Detener ejecucion por parte del usuario su pulsa "espacio"


