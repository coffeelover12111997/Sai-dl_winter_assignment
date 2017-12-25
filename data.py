#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 23 23:08:20 2017

@author: pritish
"""

import cv2
import numpy as np
import pandas as pd
import math


def dataloader(address):
    
    file=open(address)
    
    image=[]
    class_=[]
    breed=[]
    species=[]

    for i in file:
        a=i.split(' ')
        image.append(a[0])
        class_.append(a[1])
        species.append(a[2])
        breed.append(a[3])

    
    class_=pd.Series(class_)
    #class_=pd.get_dummies(class_)
    class_=np.array(class_)
    
    
    
    imval=[]

    for i in image:
        j=cv2.imread('/home/pritish/Downloads/images/'+i+'.png')
        imval.append(j)
        
    for i in range(len(imval)):
        
        if imval[i].shape[0]>imval[i].shape[1]:
            temp=imval[i]
            temp=cv2.copyMakeBorder(imval[i],math.floor((imval[i].shape[0]-imval[i].shape[1])/2),math.ceil((imval[i].shape[0]-imval[i].shape[1])/2),0,0,cv2.BORDER_CONSTANT,value=[0,0,0])
            imval[i] = cv2.resize(temp,(227, 227))
        
        if imval[i].shape[1]>imval[i].shape[0]:
            temp=imval[i]
            temp=cv2.copyMakeBorder(imval[i],0,0,math.floor((imval[i].shape[1]-imval[i].shape[0])/2),math.ceil((imval[i].shape[1]-imval[i].shape[0])/2),cv2.BORDER_CONSTANT,value=[0,0,0])
            imval[i] = cv2.resize(temp,(227, 227))
    
        if imval[i].shape[1]==imval[i].shape[0]:
            imval[i] = cv2.resize(temp,(227, 227))

    imval=np.array(imval,dtype=np.uint8)
    imval=imval.astype('float32')

    for i in range(imval.shape[0]):
        imval[i]=imval[i]*(1.0/255.0)
        imval[i]=2*imval[i]-1
    
    return imval,class_
    
  