#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  4 18:16:21 2018

@author: pritish
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 19 15:21:27 2017

@author: pritish
"""

#
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torch
from data import dataloader
import numpy as np
#import h5py
import pandas as pd
import matplotlib.pyplot as plt
import cv2

#traindata

trainX,trainy=dataloader('/home/pritish/Downloads/annotations/trainval.txt')

im=imval[100]    
#plt.imshow(im)
plt.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))



h5f = h5py.File('trim.h5','r')
trainX=h5f['dataset_2'][:]
trainy=h5f['dataset_1'][:]
h5f.close()
#testdata



testX,testy=dataloader('/home/pritish/Downloads/annotations/test.txt')
h5f = h5py.File('testim.h5','r')
testX=h5f['dataset_2'][:]
testy=h5f['dataset_1'][:]
h5f.close()




trainX=np.swapaxes(trainX,1,3).astype('float32')
trainy=np.array(trainy).astype('int64')
trainy=trainy-np.ones(trainy.shape)
'''trainy=pd.get_dummies(trainy)
trainy=np.array(trainy)

testy=pd.get_dummies(testy)
testy=np.array(testy)'''

testX=np.swapaxes(testX,1,3).astype('float32')
testy=np.array(testy).astype('int64')
testy=testy-np.ones(testy.shape)

totalsize=trainX.shape[0]



#model
epochs=3
lr=0.001
batchsize=10
    

class convnet(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.conv=nn.Sequential(nn.Conv2d(3,48,11,stride=4,padding=2),
                                nn.ReLU(),
                                nn.MaxPool2d(3,stride=2),
                                nn.Conv2d(48,96,5,stride=1,padding=1),
                                nn.ReLU(),
                                nn.MaxPool2d(3,stride=2),
                                nn.Conv2d(96,256,3,stride=1,padding=1),
                                nn.ReLU(),
                                nn.MaxPool2d(3,stride=2)
                                )
        self.linear=nn.Sequential(nn.Linear(6400,3200),
                                  nn.ReLU(),
                                  nn.Linear(3200,37)
                                  )
    
    
    def forward(self,x):
        y=self.conv.forward(x)
        
        y=y.view(-1,self.numfeatures(y))
        
        out=self.linear.forward(y)
        
        return out
    
    def numfeatures(self,x):
        num=x.size()[1:]
        i=1
        for j in num:
            i*=j
        return i


if torch.cuda.is_available():
    net=convnet()
    net=net.cuda()
else:
    net=convnet()


loss=nn.CrossEntropyLoss()

optimizer=optim.Adam(net.parameters(),lr=lr)


index=np.arange(0,totalsize)

np.random.shuffle(index)

trainX=trainX[index]
trainy=trainy[index]

net.train()
for i in range(4):
    
    for j in range(int(totalsize/batchsize)):
        #np.random.shuffle(index)
        tX,ty=trainX[index[j*batchsize:(j+1)*batchsize]],trainy[index[j*batchsize:(j+1)*batchsize]]
        tX,ty=Variable(torch.from_numpy(tX)),Variable(torch.LongTensor(ty))
        
        
        '''idxs=np.where(ty>0)[1]
        new_targets=Variable(torch.LongTensor(idxs))'''
        
        optimizer.zero_grad()
        '''if torch.cuda.is_available():
            trainX=Variable(torch.from_numpy(tX).cuda())
            new_targets=Variable(torch.LongTensor(idxs).cuda())'''
        predict=net(tX)
        
        #predict=predict.view(batchsize,-1)
       
        l=loss(predict,ty)
        l.backward()
        optimizer.step()
    print(l.data[0])
        
correct = 0
total = testX.shape[0]


for i in range(int(testX.shape[0]/batchsize)):
    predict=net(Variable(torch.from_numpy(testX[i*batchsize:(i+1)*batchsize])))
    _, predicted = torch.max(predict.data, 1)
    label=torch.LongTensor(testy[i*batchsize:(i+1)*batchsize])
    correct += (predicted == label).sum()
    

print('accuracy:'+str(correct/total))
    
    
                    
