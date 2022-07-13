## Librerias a añadir

import onnxruntime as ort

## Acciones previas a realziar una unica vez, entiendo en el apartado (__init)

#Definir path del Modelo
Model_path=r"Modelos\Comar Models\ScrewNet2.onnx"


#Definir entorno de inferencia   

ort_session = ort.InferenceSession(Model_path, providers=["CUDAExecutionProvider"])
                                    

outputs, output_names, input_name = prolocate_model(ort_session):



def _image_cb(self, msg : Image):
    if not self.enabled: return
    
    # Get the image as a numpy matrix
    img = self._cv_bridge.imgmsg_to_cv2(msg)
    
    # TODO Vision processing --> (Realizar inferencia)
    
    prediction = ort_session.run(output_names, {input_name: img}) 
    
    transform = TransformStamped()
    transform.child_frame_id = "screw"
    transform.header.frame_id = msg.header.frame_id
    # TODO Fill in the transform using vision data
    # transform.transform = ... 
    
    #   entiendo que aqui se postprocesa la salida de la red
    
    x,y = postProcessOut(prediction)
    
    self._br.sendTransform(transform)   

															   
															   
def	prolocate_model(ort_session):
    outputs = ort_session.get_outputs()
    output_names = list(map(lambda output: output.name, outputs))
    input_name = ort_session.get_inputs()[0].name
    
    
def postProcessOut(prediction):

    x=detections[0][0][0]
    y=detections[0][0][1]
    
    # Conversion a mm
    #x=
    #y=
    retunr x,y
