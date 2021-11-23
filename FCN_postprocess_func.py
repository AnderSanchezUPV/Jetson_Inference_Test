# -*- coding: utf-8 -*-
"""
Created on Tue Nov 16 16:25:21 2021

@author: ander
"""
import numpy as np
from PIL import Image
from matplotlib.colors import hsv_to_rgb
import cv2

classes = [line.rstrip('\n') for line in open('Modelos/Onnx/FCN_ResNet50/voc_classes.txt')]
num_classes = len(classes)

def get_palette():
    # prepare and return palette
    palette = [0] * num_classes * 3
    
    for hue in range(num_classes):
        if hue == 0: # Background color
            colors = (0, 0, 0)
        else:
            colors = hsv_to_rgb((hue / num_classes, 0.75, 0.75))
            
        for i in range(3):
            palette[hue * 3 + i] = int(colors[i] * 255)
            
    return palette

def colorize(labels):
    # generate colorized image from output labels and color palette
    result_img = Image.fromarray(labels).convert('P', colors=num_classes)
    result_img.putpalette(get_palette())
    return np.array(result_img.convert('RGB'))

def visualize_output(image, output):
    assert(image.shape[0] == output.shape[1] and \
           image.shape[1] == output.shape[2]) # Same height and width
    assert(output.shape[0] == num_classes)
    
    # get classification labels
    raw_labels = np.argmax(output, axis=0).astype(np.uint8)

    # comput confidence score
    confidence = float(np.max(output, axis=0).mean())

    # generate segmented image
    result_img = colorize(raw_labels)
    
    # generate blended image
    blended_img = cv2.addWeighted(image[:, :, ::-1], 0.5, result_img, 0.5, 0)
    
    # comentar para poder operar con OpenCV
    # result_img = Image.fromarray(result_img)
    # blended_img = Image.fromarray(blended_img)

    return confidence, result_img, blended_img, raw_labels