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
import numpy as np
import h5py


#traindata
h5f = h5py.File('trim.h5','r')
trainX=h5f['dataset_2'][:]
trainy=h5f['dataset_1'][:]
h5f.close()
#testdata
h5f = h5py.File('teim.h5','r')
testX=h5f['dataset_2'][:]
testy=h5f['dataset_1'][:]
h5f.close()

trainX=np.swapaxes(trainX,1,3).astype('float32')

testX=np.swapaxes(testX,1,3).astype('float32')

totalsize=trainX.shape[0]

#model
epochs=15
lr=0.001
batchsize=10
    

class convnet(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.conv1=nn.Conv2d(3,48,11,stride=4,padding=2)
        self.c1b=nn.BatchNorm2d(48)
        self.conv2=nn.Conv2d(48,96,5,stride=1,padding=1)
        self.c2b=nn.BatchNorm2d(96)
        self.conv5=nn.Conv2d(96,128,3,stride=1,padding=1)
        self.c5b=nn.BatchNorm2d(128)
        self.pool=nn.MaxPool2d(3,stride=2)
        self.fc1=nn.Linear(3200,1600)
        self.fb1=nn.BatchNorm1d(1600)
        #self.fc2=nn.Linear(1600,1600)
        self.fc3=nn.Linear(1600,37)
        self.re=nn.ReLU()
        self.drop=torch.nn.modules.Dropout(p=0.5)
    
    
    def forward(self,x):
        x=self.pool(self.re(self.c1b(self.conv1(x))))
        x=self.pool(self.re(self.c2b(self.conv2(x))))
        #x=F.relu(self.conv3(x))
        #x=F.relu(self.conv4(x))
        x=self.pool(self.re(self.c5b(self.conv5(x))))
        
        x=x.view(-1,self.numfeatures(x))
        x=self.re(self.fb1(self.fc1.forward(x)))
        #x=F.relu(self.fc2(x))
        x=self.drop(x)
        out=self.fc3.forward(x)
        
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

net.train()
for i in range(epochs):
    for j in range(int(totalsize/batchsize)):
        tX,ty=trainX[index[j*batchsize:(j+1)*batchsize]],trainy[index[j*batchsize:(j+1)*batchsize]]
        tX,ty=Variable(torch.from_numpy(tX)),Variable(torch.LongTensor(ty))
        optimizer.zero_grad()
        '''if torch.cuda.is_available():
            trainX=Variable(torch.from_numpy(tX).cuda())
            new_targets=Variable(torch.LongTensor(idxs).cuda())'''
        predict=net(tX)
        l=loss(predict,ty)
        l.backward()
        optimizer.step()
    print(l.data[0])
        
correct = 0
total = testX.shape[0]

net.eval()
for i in range(int(testX.shape[0]/batchsize)):
    predict=net(Variable(torch.from_numpy(testX[i*batchsize:(i+1)*batchsize])))
    _, predicted = torch.max(predict.data, 1)
    label=torch.LongTensor(testy[i*batchsize:(i+1)*batchsize])
    correct += (predicted == label).sum()
    

print('accuracy:'+str(correct/total))
    
    
                    
