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
import os

import cv2
from numpy.lib.stride_tricks import DummyArray

import engine as eng
import Inference as inf
import tensorrt as trt 

from  YoloV4_postprocess_func import *




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

input_size = 416

##  Incio del programa
print("Inicializando Programa")

##  Definir parametros para el Para el Postprocesado
print("Definir parametros de postprocesado")
ANCHORS = "Modelos/YoloV4/yoloV4_anchors.txt"
STRIDES = [8, 16, 32]
XYSCALE = [1.2, 1.1, 1.05]

ANCHORS = get_anchors(ANCHORS)
STRIDES = np.array(STRIDES)

##  Definir objeto de la camara
print("Generar Objeto de la camara")
if os.name=='nt':
    cam = cv2.VideoCapture(0)   # Windows
elif os.name =='posix':    
    #cam = cv2.VideoCapture('v4l2src device=/dev/video0 ! jpegdec ! videoconvert  ! video/x-raw, width=640, height=480 ! appsink',cv2.CAP_GSTREAMER)
    cam = cv2.VideoCapture('v4l2src device=/dev/video0 ! jpegdec ! videoconvert  ! video/x-raw, width=640, height=480 ! appsink drop=true sync=false',cv2.CAP_GSTREAMER)
    
else:
    print("Error al crear objeto de la camara")  
    

## Propiedades del texto en pantalla
font                   = cv2.FONT_HERSHEY_SIMPLEX
bottomLeftCornerOfText = (2,475)
fontScale              =  0.7
fontColor              = (255,255,255)
lineThickness          = 1

##  Definir parametros de la Inferencia
print("Definir modelo y entorno de ejecucion")
model_dir ="./ToTensor/convert_YoloV4"
serialized_plan_fp32 = model_dir+"/YoloV4.plan"
HEIGHT = 416
WIDTH = 416
variablesTensor = trt.float32
VarialbesNumpy = np.float32

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
trt_runtime = trt.Runtime(TRT_LOGGER)

##	Cargar Modelo en formato .plan
engine = eng.load_engine(trt_runtime, serialized_plan_fp32)
h_input, d_input, h_output, d_output, stream = inf.allocate_buffers(engine, 1, variablesTensor)



print("Lazo principal")
print("####################################")
while True:
    try:
        model_start=time.time()
        #Capturar imagen Desde camara 
        #print("Tomar Imagen")
        cv_flag ,original_image = cam.read() 
        
        if not(cv_flag):
            print("Error en la toma de imagen")
            cam.release()
            cv2.destroyAllWindows()
            break
           
        ##  Preprocesar Imagen
        #print("Preprocesar Imagen")
       
        
        # original_image_size = original_image.shape[:2]
        original_image_size=np.shape(original_image)[:2]
        
        image_data = image_preprocess(np.copy(original_image), [input_size, input_size])
        
        image_data = image_data[np.newaxis, ...].astype(np.float32)
        
        #print("Preprocessed image shape:",image_data.shape) # shape of the preprocessed inp
        
        
        ##  Inferencia    
        #print("Inferencia")
        Inference_start=time.time()
        out = inf.do_inference(engine, image_data, h_input, d_input, h_output, d_output, stream, 1,HEIGHT, WIDTH)        
        # detections = sess.run(output_names, {input_name: image_data})
        #print("Output shape:", list(map(lambda detection: detection.shape, detections)))
        Inference_end=time.time()-Inference_start
        
        ##  PostProcesar Imagen
        #print("Postprocesar Imagen")
        #   Las funciones de postprocesado se importan desde: YoloV4_postprocess_func
        
        # pred_bbox = postprocess_bbbox(detections, ANCHORS, STRIDES, XYSCALE)
        # bboxes = postprocess_boxes(pred_bbox, original_image_size, input_size, 0.25)
        bboxes = nms(bboxes, 0.213, method='nms')
        image = draw_bbox(original_image, bboxes)  
        
        model_end=time.time()-model_start

        #print("Tiempo de inferencia {:.4f} ms".format(model_end*1000))
        ##  Mostrar Resultado
        #print("Mostrar Imagen")
        image_text='Tiempo de Inferencia: {:.2f} ms   Tasa de FP: {:.2f}: '.format(Inference_end*1000,1/model_end)
        
        image=cv2.putText(image,image_text,
                    bottomLeftCornerOfText, 
                    font, 
                    fontScale,
                    fontColor,
                    lineThickness)
        
        cv2.imshow('YoloV4 Output',image)
        k=cv2.waitKey(1)        
        ##  Detener ejecucion por parte del usuario su pulsa "espacio"
        if k%256==32:
            print('Interrumpido por el Usuario \n\n ## Fin del programa ##')
            break
        
        
        # print("Tamaño de la imagen de salida: {}".format(np.shape(image)[:2]))
        
    except:
        cam.release()
        cv2.destroyAllWindows()
        print("Excepcion!!!")
        break
    
##  Finalizarla Ejecucion del programa
cam.release()
cv2.destroyAllWindows()
