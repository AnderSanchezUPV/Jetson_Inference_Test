# -*- coding: utf-8 -*-
"""
Created on Thu Nov 18 11:51:15 2021

@author: ander
"""

##  Importar Librerias
import numpy as np
# import glob
import os
# import onnx
import onnxruntime as ort
import time
import numpy
import cv2 

# from onnx import numpy_helper
from PIL import Image

import torchvision.transforms as transforms
to_tensor = transforms.ToTensor()

from FCN_postprocess_func import *

##  Definiciones
preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])




##  Incio del programa
print("Inicializando Programa")

## Propiedades del texto en pantalla
font                   = cv2.FONT_HERSHEY_SIMPLEX
bottomLeftCornerOfText = (2,475)
fontScale              =  0.4 
fontColor              = (255,255,255)
lineThickness          = 1

##  Definir objeto de la camara
print("Generar Objeto de la camara")
if os.name=='nt':
    cam = cv2.VideoCapture(0)   # Windows
elif os.name =='posix':
    cam = cv2.VideoCapture(0,cv2.CAP_V4L) # Linux
else:
    print("Error al crear objeto de la camara")  

##  Definir parametros de la Inferencia
print("Definir modelo y entorno de ejecucion")
Model_path="Modelos/FCN_ResNet50/fcn-resnet50-11.onnx"

ort_session = ort.InferenceSession(Model_path,
                                     providers=["CUDAExecutionProvider"])

#ort_session = ort.InferenceSession(Model_path,
#                                    providers=["CPUExecutionProvider"])
print("Lazo principal")
print("####################################")
while True:
    try:
        model_start=time.time()
        #Capturar imagen Desde camara 
        print("Tomar Imagen")
        cv_flag ,original_image = cam.read()
        if not(cv_flag):
            print("Error en la toma de imagen")
            cam.release()
            cv2.destroyAllWindows()
            break
           
        ##  Preprocesar Imagen
        print("Preprocesar Imagen")
        orig_tensor = np.asarray(original_image)
        orig_tensor = cv2.cvtColor(orig_tensor, cv2.COLOR_BGR2RGB)
        img_data = preprocess(original_image)
        img_data = img_data.unsqueeze(0)
        img_data = img_data.detach().cpu().numpy()
        
        #print("Preprocessed image shape:",image_data.shape) # shape of the preprocessed inp        
        
        ##  Inferencia    
        print("Inferencia")
        inference_start=time.time()
        ort_inputs={ort_session.get_inputs()[0].name: (img_data)}#,
                        # ort_session.get_inputs()[1].name: image_size}
        ort_outs = ort_session.run(None, ort_inputs)

        output, aux = ort_outs

        # print("Output shape:", list(map(lambda detection: detection.shape, ort_outs)))
          
        inference_time=time.time()-inference_start
            
        print("Tiempo de inferencia {:.4f} ms".format(inference_time*1000))
        
        ##  PostProcesar Imagen
        print("Postprocesar Imagen")        
        
        conf, result_img, blended_img, raw_labels = visualize_output(orig_tensor, output[0])
        
        # print("Tiempo de inferencia {:.4f} ms".format(model_end*1000))
        ##  Mostrar Resultado
        print("Mostrar Imagen")
        
        model_end=time.time()-model_start
        image_text='Tiempo de Ejecucion: {:.2f} ms  TIempo de inferencia: {:.2f} ms'.format(model_end*1000,inference_time*1000)
        
        blended_img=cv2.putText(blended_img,image_text,
                    bottomLeftCornerOfText, 
                    font, 
                    fontScale,
                    fontColor,
                    lineThickness)
        
        cv2.imshow('Superposicion',blended_img)
        k=cv2.waitKey(1)             
              
        ##  Detener ejecucion por parte del usuario su pulsa "espacio"
        if k%256==32:
            print('Interrumpido por el Usuario \n\n ## Fin del programa ##')
            break
        model_end=time.time()-model_start
        
    except:
        cam.release()
        cv2.destroyAllWindows()
        print("Excepcion!!!")
        break
    
##  Finalizarla Ejecucion del programa
cam.release()
cv2.destroyAllWindows()
