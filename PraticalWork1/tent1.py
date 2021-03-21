# -*- coding: utf-8 -*-
"""
Created on Mon Mar 15 16:51:37 2021

@author: pedrk
"""
import cv2 as cv
import glob
import os



img_dir = "Imagens/" # Enter Directory of all images  
data_path = os.path.join(img_dir,'*jpg') 
files = glob.glob(data_path) 
data = [] 

# pasta para guardar os resultados do otsu
directory="Otsu"
path_dir="C:/Users/pedrk/Desktop/Visão por Computador/TP1/"
path=os.path.join(path_dir,directory)
os.mkdir(path)

d=0

for f1 in files: 
    # Metodo de Otsu
    img = cv.imread(f1) 
    data.append(img)
    # cv.imshow('asd',img)

    # converter a imagem para grayscale
    gray_img2=cv.cvtColor(img,cv.COLOR_BGR2GRAY)
    # cv.imshow('gray image2',gray_img2)

    ret2,img_seg = cv.threshold(gray_img2,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
    cv.imshow('Segmentacao',img_seg)
    
    filename="Otsu/img_%d.jpg"%d
    cv.imwrite(filename,img_seg)
    d+=1
    
    # Metodo
    

cv.waitKey(0) 
    


