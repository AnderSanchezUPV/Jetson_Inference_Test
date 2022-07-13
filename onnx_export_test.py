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
import time


## Definir Arrays para emular imagenes.
Patron=np.zeros((1,1,80,80),dtype=np.float32)
Imagen=np.zeros((1,1,512,612),dtype=np.float32)

##  Cargar Modelo
#Model_path=r"Modelos\Comar Models\CoMAr_CNN_V02.onnx"
Model_path=r"Modelos/Comar Models/ScrewNet.onnx"

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
#input_name1 = ort_session.get_inputs()[1].name

##  Inferencia

n_sim=500
time_array=np.zeros(n_sim)
for i in range (n_sim): 

    inference_start=time.time()


    #
    #detections = ort_session.run(output_names, {input_name0: Imagen, input_name1: Patron})
    detections = ort_session.run(output_names, {input_name0: Imagen}) 
    inference_time=time.time()-inference_start
    
    print("Tiempo de inferencia {:.4f} ms".format(inference_time*1000))
    #print(detections)
    time_array[i]=inference_time
    
worst_time=np.amax(time_array[1:n_sim])  
mean_time=np.mean(time_array[1:n_sim])       
print("Tiempo medio {:.4f} ms   Peor Tiempo= {:.4f} ms".format(mean_time*1000,
                                                               worst_time*1000))