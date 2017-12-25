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
import torch.nn.functional as F
import torch
from data import dataloader
import numpy as np

#traindata

imval,class_=dataloader('/home/pritish/Downloads/annotations/trainval.txt')

#testdata

imvalt,classt=dataloader('/home/pritish/Downloads/annotations/test.txt')

import pandas as pd
classt=pd.get_dummies(classt)
classt=np.array(classt)

X=imval
X=np.swapaxes(X,1,3)
y=class_

y=pd.get_dummies(y)
y=np.array(y)

imvalt=np.swapaxes(imvalt,1,3)
labels=classt

#model
epochs=3
lr=0.001
batchsize=10
    
class convnet(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.conv1=nn.Conv2d(3,48,11,4,2)
        self.conv2=nn.Conv2d(48,128,5,1,1)
        #self.conv3=nn.Conv2d(128,192,3,1,1)
        #self.conv4=nn.Conv2d(192,192,3,1,1)
        self.conv5=nn.Conv2d(128,256,3,1,1)
        self.pool=nn.MaxPool2d(3,2)
        self.fc1=nn.Linear(6400,3200)
        #self.fc2=nn.Linear(3200,3200)
        self.fc3=nn.Linear(3200,37)
    
    def forward(self,x):
        x=self.pool(F.relu(self.conv1(x)))
        x=self.pool(F.relu(self.conv2(x)))
        #x=F.relu(self.conv3(x))
        #x=F.relu(self.conv4(x))
        x=self.pool(F.relu(self.conv5(x)))
        
        x=x.view(-1,self.numfeatures(x))
        x=F.relu(self.fc1(x))
        x=torch.nn.modules.Dropout(x)
        #x=F.relu(self.fc2(x))
        out=self.fc3(x)
        
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

totalsize=X.shape[0]
index=np.arange(0,X.shape[0])

for i in range(epochs):
    for j in range(int(totalsize/batchsize)):
        np.random.shuffle(index)
        tX,ty=X[index[:batchsize]],y[index[:batchsize]]
        trainX=Variable(torch.from_numpy(tX))
        idxs=np.where(ty>0)[1]
        new_targets=Variable(torch.LongTensor(idxs))
        if torch.cuda.is_available():
            trainX=trainX.cuda()
            new_targets=new_targets.cuda()
        predict=net(trainX)
        l=loss(predict,new_targets)
        l.backward()
        optimizer.step()
    #print(l)
        
correct = 0
total = imvalt.shape[0]

for i in range(imvalt.shape[0]):
    predict=net(Variable(torch.from_numpy(imvalt[i].reshape(-1,3,227,227))))
    _, predicted = torch.max(predict.data, 1)
    correct += (predicted == torch.LongTensor(labels[i])).sum()

print('accuracy:'+str(correct/total))
    
    
                    