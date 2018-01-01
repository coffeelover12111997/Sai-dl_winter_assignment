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
#from data import dataloader
import numpy as np
import h5py
import pandas as pd

#traindata

#imval,class_=dataloader('/home/pritish/Downloads/annotations/trainval.txt')
h5f = h5py.File('trainim.h5','r')
trainX=h5f['dataset_2'][:]
trainy=h5f['dataset_1'][:]
h5f.close()
#testdata



#imvalt,classt=dataloader('/home/pritish/Downloads/annotations/test.txt')
h5f = h5py.File('testim.h5','r')
testX=h5f['dataset_2'][:]
testy=h5f['dataset_1'][:]
h5f.close()



trainX=np.swapaxes(trainX,1,3)
trainy=trainy-np.ones(trainy.shape)
'''trainy=pd.get_dummies(trainy)
trainy=np.array(trainy)

testy=pd.get_dummies(testy)
testy=np.array(testy)'''

testX=np.swapaxes(testX,1,3)
testy=testy-np.ones(testy.shape)

totalsize=trainX.shape[0]



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
        self.conv5=nn.Conv2d(128,128,3,1,1)
        self.pool=nn.MaxPool2d(3,2)
        self.fc1=nn.Linear(3200,1600)
        self.fc2=nn.Linear(1600,1600)
        self.fc3=nn.Linear(1600,37)
        self.drop=torch.nn.modules.Dropout(p=0.5)
    
    def forward(self,x):
        x=self.pool(F.relu(self.conv1(x)))
        x=self.pool(F.relu(self.conv2(x)))
        #x=F.relu(self.conv3(x))
        #x=F.relu(self.conv4(x))
        x=self.pool(F.relu(self.conv5(x)))
        
        x=x.view(-1,self.numfeatures(x))
        x=F.relu(self.fc1(x))
        x=F.relu(self.fc2(x))
        x=self.drop(x)
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


index=np.arange(0,totalsize)

np.random.shuffle(index)

trainX=trainX[index]
trainy=trainy[index]

for i in range(3):
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
    print(l)
        
correct = 0
total = testX.shape[0]

for i in range(int(testX.shape[0]/batchsize)):
    predict=net(Variable(torch.from_numpy(testX[100*batchsize:(101)*batchsize])))
    _, predicted = torch.max(predict.data, 1)
    label=torch.LongTensor(trainy[i*batchsize:(i+1)*batchsize])
    correct += (predicted == label).sum()
    

print('accuracy:'+str(correct/total))
    
    
                    
