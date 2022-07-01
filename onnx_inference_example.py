# -*- coding: utf-8 -*-
"""
Created on Wed Nov 10 16:34:57 2021

@author: ander

Onnx Export Test
"""

## Librerias
import numpy as np
import onnx
import onnxruntime as ort



## Definir Arrays para emular imagenes.
Patron=np.zeros((1,1,80,80),dtype=np.float32)
Imagen=np.zeros((1,1,512,612),dtype=np.float32)

##  Cargar Modelo
Model_path="Modelos\Comar Models\inf_time_test.onnx"


##  Definir entorno de inferencia   
#   - Path al modelo
#   - Ejecutar en Grafica (CUDA) o en procesador
#   - Para (CUDA), en caso de varias graficas, seleccionar el dispositivo de jecucion
ort_session = ort.InferenceSession(Model_path, 
                                    providers=["CUDAExecutionProvider"],
                                    provider_options=[{"device_id": 0}])

# ort_session = ort.InferenceSession(Model_path,
#                                     providers=["CPUExecutionProvider"])


#   Definir los nombres de las variables y entradas de salida para la inferencia. 
#   Se extraen de la informacion del modelo
outputs = ort_session.get_outputs()
output_names = list(map(lambda output: output.name, outputs))
input_name0 = ort_session.get_inputs()[0].name
input_name1 = ort_session.get_inputs()[1].name

#   placeholder para la funcion de psotprocesado de la salida de la red
def postprocess_out(detections):
    X=detections[0]
    Y=detections[1]
    Theta=detections[2]
    return X,Y,Theta




##  Inferencia

n_sim=500

for i in range (n_sim): 


    # Instruccion para la ejecucion de inferencia, detections devuelve un array 1x3 con la respues
    # en bruto de la red

    detections = ort_session.run(output_names, {input_name0: Imagen, input_name1: Patron})

    X,Y,Theta=postprocess_out(detections)