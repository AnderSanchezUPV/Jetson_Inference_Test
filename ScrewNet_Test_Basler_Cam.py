#ScrewNet_Test_Basler_Cam
from pypylon import pylon
import numpy as np
import cv2
import pickle
import Pose_stimation_functions as Ps
import os
from datetime import date, datetime
import time
import onnxruntime as ort


                        ####        Set up Inference Enviroment ####
Model_path=r"Modelos/Comar Models/ScrewNet.onnx"      

ort_session = ort.InferenceSession( Model_path, 
                                    providers=["CUDAExecutionProvider"])

outputs = ort_session.get_outputs()
output_names = list(map(lambda output: output.name, outputs))
input_name0 = ort_session.get_inputs()[0].name

                        ####        Conexion camera GIGE        ####
# conecting to the first available camera
camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())

# Grabing Continusely (video) with minimal delay
camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly) 
converter = pylon.ImageFormatConverter()

# converting to opencv bgr format
converter.OutputPixelFormat = pylon.PixelType_BGR8packed
converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned


while camera.IsGrabbing():
    try:
        grabResult = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)

        if grabResult.GrabSucceeded():
            # Access the image data
            image = converter.Convert(grabResult)
            img = image.GetArray()
            org_img=np.copy(img)

           #  Inference
            detections = ort_session.run(output_names, {input_name0: org_img})


            cv2.imshow('title', org_img)
            if cv2.waitKey(1) & 0xFF == 32:
                camera.StopGrabbing()
                cv2.destroyAllWindows()
                break
            
        time.sleep(0.250)

    except:
        print('Error en loop principal')
        camera.StopGrabbing()
        cv2.destroyAllWindows()        
        break