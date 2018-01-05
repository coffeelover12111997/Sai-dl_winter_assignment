#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 26 14:40:39 2017

@author: pritish
"""

#asin(wx+p)

import numpy as np
import h5py



train=[]
test=[]

a=np.arange(0,2,0.1)
X=np.arange(0,20,0.1)
fi=(np.arange(0,180,10))*(np.pi/180)

for i in range(5000):
    A=np.random.choice(a,size=1)
    w=np.random.choice(X[1:],size=1)
    p=np.random.choice(fi,size=1)
    W=(2*(np.pi))/w
    train.append(A*(np.sin((W*X)+p)))
    
for i in range(2000):
    A=np.random.choice(a,size=1)
    w=np.random.choice(X[1:],size=1)
    p=np.random.choice(fi,size=1)
    W=(2*(np.pi))/w
    test.append(A*(np.sin((W*X)+p)))
t=[]
te=[]
trainnoisy=[]
testnoisy=[]

for i in train:
        noise=(0.3*(np.max(i))*np.random.rand(200))
        trainnoisy.append(i+noise)
        t.append(i)

for i in test:
        te.append(i)
        noise=(0.3*(np.max(i))*np.random.rand(200))
        testnoisy.append(i+noise)

t=np.array(t)
te=np.array(te)
trainnoisy=np.array(trainnoisy)
testnoisy=np.array(testnoisy)

hf=h5py.File('trainsine.h5','w')
hf.create_dataset('dataset_1', data=t)
hf.create_dataset('dataset_2', data=trainnoisy)
hf.close()

hf=h5py.File('testsine.h5','w')
hf.create_dataset('dataset_1', data=te)
hf.create_dataset('dataset_2', data=testnoisy)
hf.close()


'''end'''


'''test the signals'''



h5f = h5py.File('trainsine.h5','r')
b=h5f['dataset_1'][:]
c=h5f['dataset_2'][:]
h5f.close()





import matplotlib.pyplot as plt
X=np.arange(0,20,0.1) 
plt.plot(X, trainnoisy[10])
plt.xlabel('Angle [rad]')
plt.ylabel('sin(x)')
plt.axis('tight')


