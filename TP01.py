# -*- coding: utf-8 -*-
"""
Created on Mon Mar 15 18:30:47 2021

@author: pedrk
"""
import cv2 as cv

img=cv.imread('Imagens/3096.jpg')
cv.imshow('aviao',img)

# converter a imagem para grayscale
# gray_img=rgb2gray(img)
# cv.imshow('gray image',gray_img)
gray_img2=cv.cvtColor(img,cv.COLOR_BGR2GRAY)
cv.imshow('gray image2',gray_img2)

# th2 = cv.adaptiveThreshold(gray_img2,255,cv.ADAPTIVE_THRESH_MEAN_C,cv.THRESH_BINARY,505,55)
# cv.imshow('Binarizacao',th2)

ret2,th2 = cv.threshold(gray_img2,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
cv.imshow('Segmentacao',th2)

cv.waitKey(0) 