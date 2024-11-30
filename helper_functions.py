import os
import cv2
import PIL.Image as Image
import numpy as np


def img_gray(file):
    image = cv2.imread(file)
    image = cv2.resize(image, [640, 480])
    
    image_copy = image.copy()
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5,5), 0)
    
    return gray, image_copy

    
def binary_mask(gray_image):
    _, mask = cv2.threshold(gray_image, 254, 255, cv2.THRESH_BINARY_INV)
    
    return mask