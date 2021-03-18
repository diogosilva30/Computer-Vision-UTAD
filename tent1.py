# -*- coding: utf-8 -*-
"""
Created on Mon Mar 15 16:51:37 2021

@author: pedrk
"""
import cv2 as cv
import glob
import os

# Metodo de Otsu
img_dir = "Imagens/" # Enter Directory of all images  
data_path = os.path.join(img_dir,'*jpg') 
files = glob.glob(data_path) 
data = [] 
for f1 in files: 
    img = cv.imread(f1) 
    data.append(img) 

    # converter a imagem para grayscale
    # gray_img2=cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    # cv.imshow('gray image2',gray_img2)

    # ret2,th2 = cv.threshold(gray_img2,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
    # cv.imshow('Segmentacao',th2)

    cv.waitKey(0) 
    


      