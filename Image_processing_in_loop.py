# -*- coding: utf-8 -*-
"""
Created on Fri Oct 22 15:51:56 2021

@author: Ander Sanchez

Este programa accede a la webcam del ordenador para toar imagenes de forma 
iterativa.
Dicha imagen es procesada por la red resnet-50
Se muestra por pantalla la clase mas relevante detectada por la red.

"""
## Limpiar el workspace
#from IPython import get_ipython
#get_ipython().magic('reset -sf')

## Importar librerias
import cv2 
import time
import torch
import torchvision.transforms as transforms
from torchvision.transforms.functional import to_tensor
import onnx
import onnxruntime as ort
import numpy as np
from PIL import Image


import keyboard

##  Definiciones
def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

def preprocess(img):    
    sq_transformsFc= torch.nn.Sequential(
        transforms.Resize([224, 224]),
        transforms.CenterCrop(224),
        # transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])        
        )
    
    img=sq_transformsFc(img)
    return (img)

# scripted_process= torch.jit.script(transforms)

def preprocess_pil(img):    
    sq_transformsFc= transforms.Compose([
        transforms.Resize([224, 224]),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])        
        ])
    
    img=sq_transformsFc(img)
    return (img)

##  hook event handleer del teclado
stop = False


##  Obtener lsita de etiquetas de Imagenet
text_file = open("Modelos\resnet50v2\imagenet1000_clsidx_to_labels.txt", "r")
Img_net_labels = text_file.read().splitlines()
text_file.close()

##  Crear objeto de la camara
cam = cv2.VideoCapture(0)  
# cv2.namedWindow('cam-test')

## Propiedades del texto en pantalla
font                   = cv2.FONT_HERSHEY_SIMPLEX
bottomLeftCornerOfText = (2,475)
fontScale              =  0.4 
fontColor              = (255,255,255)
lineThickness          = 1

##  Loop toma imagen
idx=1;
current_tic=time.time()
##  Definir Modelo neuronal
#ort_session = ort.InferenceSession("Modelos/resnet50v2/resnet50-v2-7.onnx",
#                                      providers=["CPUExecutionProvider"])
ort_session = ort.InferenceSession("Modelos/resnet50v2/resnet50-v2-7.onnx",
                                       providers=["CUDAExecutionProvider"])
## Main loop
print('Main Loop')
while not(stop):   
    try:
        previous_tic=current_tic
        current_tic=time.time()
        #  Leer y comprobar imagen  
        cv_flag ,img = cam.read() 
        if not(cv_flag):  
            print('Error al capturar imagen')
            break
        # Preprocesar imagen
        preprocessing_start=time.time();            
        img_net=preprocess_pil(Image.fromarray(img))        
        #img_net=preprocess(to_tensor(img))       
 
        
        img_net.unsqueeze_(0)
        preprocesing_time=time.time()-preprocessing_start
        # print('Tiempo de procesado: {}'.format(preprocesing_time))
        
        # Inferencia
        
        inference_start=time.time();
        ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(img_net)}
        ort_outs = ort_session.run(None, ort_inputs)
        prediction=np.argmax(ort_outs[0])
        inference_time=time.time()-inference_start;
        
        # print('Tiempo de inferencia: {}'.format(inference_time))
        
        #   Definir texto en pantalla
        
        ex_time=current_tic-previous_tic;
        # image_text='FPS: {}'.format(1/ex_time)
        image_text='Prediction--> {}; Ciclos por segundo: {:.2f} Tiempo de Inferencia: {:.4f}'.format(
                    Img_net_labels[prediction],1/ex_time,inference_time)
        # print('Tiempo de ejecucion: {}'.format(1/ex_time))                                               
        # ts=time.time()-ti;
        img=cv2.putText(img,image_text,
                    bottomLeftCornerOfText, 
                    font, 
                    fontScale,
                    fontColor,
                    lineThickness)
        
        #   Mostrar Imagen por pantalla
        cv2.imshow('cam-test',img)
        k=cv2.waitKey(1)    # Necesario para que el prgrama no pete. Se indica el 
                            # tiempo en ms que se bloquea la imagen en la
                            # ventana
        if k%256==32:
            print('Espace pressed \nFin del programa')
            break          
      
        
        
    except: # En caso de fallo cerrar conexion a webcam y ventanas
        print('Error en loop principal')
        cam.release()
        cv2.destroyAllWindows()
        
        break
    
## Al finalizar el programa cerrar conexion a webcam y ventanas
cam.release()
cv2.destroyAllWindows()



           
        
