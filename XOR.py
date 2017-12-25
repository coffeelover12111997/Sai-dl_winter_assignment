#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 21 16:30:24 2017

@author: pritish
"""
#xor using numpy

import numpy as np

X=np.array([1,0,0,0],dtype=float)
y=np.array([1,0,1,0],dtype=float)

def AND(X,y):
    z=np.concatenate((X,y),axis=1)
    weight=np.array([20,20],dtype=float)
    out=z.dot(np.transpose(weight.reshape(1,-1)))-30*np.ones((z.shape[0],1))
    return (out>0).astype(float)

def OR(X,Y):
    return 1-AND(1-X,1-Y)


def XOR(X,y):
    return OR(AND(X,1-y),AND(1-X,y))

print(XOR(X.reshape(-1,1),y.reshape(-1,1)))