# -*- coding: utf-8 -*-
"""
Created on Sun Dec  2 17:07:27 2018

@author: shreyasbhanderi
"""
import numpy as np
import cv2
def get_otsu_threshold(image):
    print(image.shape)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.bitwise_not(gray)
    thresh = cv2.threshold(gray, 0, 255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    return thresh
def align_image(thresh):
    #thresh = get_otsu_threshold(image)
    shape = thresh.shape
    zeros = np.zeros((thresh.shape[0], 500))
    thresh = np.hstack([zeros,thresh,zeros])
    shape = thresh.shape
    zeros = np.zeros((500, thresh.shape[1]))
    thresh = np.vstack([zeros,thresh,zeros])
    #show(thresh)
    coords = np.column_stack(np.where(thresh.T > 0))
    #print(coords.shape)
    rows,cols = thresh.shape[:2]
    [vx,vy,x,y] = cv2.fitLine(coords, cv2.DIST_WELSCH,0,0.01,0.1)
    lefty = int((-x*vy/vx) + y)
    righty = int(((cols-x)*vy/vx)+y)
    #cv2.line(thresh,(cols-1,righty),(0,lefty),(255,255,255),10)
    angle = (vy/vx)*180/3.14
    (h, w) = thresh.shape
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(thresh, M, (w, h),flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    #rotated = imutils.rotate(thresh, -angle)
    return rotated
def crop_signature_fast(image):
    #gray = get_otsu_threshold(image)
    h,w = image.shape
    xmin = 0
    xmax = w-1
    ymin = 0 
    ymax = h-1
    for i in range(w):
        if np.sum(image[:,i]):
            xmin = i
            break
            
    for i in range(w-1, 0, -1):
        if np.sum(image[:,i]):
            xmax = i
            break
            
    for i in range(h-1, 0, -1):
        if np.sum(image[i]):
            ymax = i
            break
            
    for i in range(h):
        if np.sum(image[i]):
            ymin = i
            break
            
    crop_sig = image[ymin:ymax , xmin:xmax]
    return crop_sig
def pad(img):
    new_img = np.zeros((150,550))
    if img.shape[0] == 140:
        k1 = int((550-img.shape[1])/2)
        k2 = int((550-img.shape[1])/2 + img.shape[1]%2)
        new_img[5:-5,k1:-k2] = img
    else:
        k1 = int((150-img.shape[0])/2)
        k2 = int((150-img.shape[0])/2 + img.shape[0]%2)
        new_img[k1:-k2,5:-5] = img
    return new_img
def resize(img):
    p1 = img.shape[0]/140
    p2 = img.shape[1]/540
    if p1>p2:
        p2 = int(img.shape[1]/p1)
        p1 = 140
    else:
        p1 = int(img.shape[0]/p2)
        p2 = 540
    resized = cv2.resize(img, (p2,p1), interpolation = cv2.INTER_AREA)
    resized = pad(resized)
    return resized
def process(img):
    img = get_otsu_threshold(img)
    img = align_image(img)
    img = crop_signature_fast(img)
    img = resize(img)
    return img