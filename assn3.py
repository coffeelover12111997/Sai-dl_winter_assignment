#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  5 20:37:23 2018

@author: pritish
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 26 16:55:38 2017

@author: pritish
"""

#denoising autoencoder

#modules
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np 
import h5py
import matplotlib.pyplot as plt





#load data

h5f = h5py.File('trainsine.h5','r')
trainX=h5f['dataset_2'][:]
trainy=h5f['dataset_1'][:]
h5f.close()

h5f = h5py.File('testsine.h5','r')
testX=h5f['dataset_2'][:]
testy=h5f['dataset_1'][:]
h5f.close()

def noiseadd(clean):
    output=[]
    for i in clean:
        noise=(0.3*(np.max(i))*np.random.rand(clean.shape[1]))
        output.append(i+noise)
    return np.array(output)

#model



class encoder(nn.Module):
    
    def __init__(self,n1,n2):
        super().__init__()
        self.layer1=nn.Linear(n1,n2)
        self.drop=nn.Dropout(0.2)
        self.ob=torch.rand(n1)
        
    def forward(self,x):
        y=(F.sigmoid(self.layer1(x)))
        temp=self.layer1.weight.transpose(0,1)
        t1=self.drop(y)
        z=(F.linear(t1,temp,Variable(self.ob)))
        return z,y
       
        
class decoder(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.layer1=nn.Linear(1800,600)
        self.layer2=nn.Linear(600,200)
        self.drop=nn.Dropout(0.2)
    
    def forward(self,x):
        y=(F.sigmoid(self.layer1(x)))
        t1=self.drop(y)
        z=self.layer2(t1)
        return z



enc1=encoder(200,600)

loss=nn.MSELoss()  
optimizer1=optim.Adam(enc1.parameters(),lr=0.001)


index=np.arange(0,5000)
epochs=20
batchsize=5



for i in range(epochs):
    for j in range(int(5000/batchsize)):
        tx,ty=Variable(torch.FloatTensor(trainX[j*batchsize:(j+1)*batchsize])),Variable(torch.FloatTensor(trainy[j*batchsize:(j+1)*batchsize]))
        optimizer1.zero_grad()
        predict,_=enc1(tx)
        l=loss(predict,ty)
        l.backward()
        optimizer1.step()
    print(l.data[0])
        
teX,tey=Variable(torch.FloatTensor(trainX)),Variable(torch.FloatTensor(trainy))
_,encoded1=enc1(teX)

encoded1=(encoded1.data).numpy()



enc2=encoder(600,1800)
optimizer2=optim.Adam(enc2.parameters(),lr=0.001)
n_enc1=noiseadd(encoded1)

for i in range(20):
    for j in range(int(5000/batchsize)):
        tx,ty=Variable(torch.FloatTensor(n_enc1[j*batchsize:(j+1)*batchsize])),Variable(torch.FloatTensor(encoded1[j*batchsize:(j+1)*batchsize]))
        optimizer2.zero_grad()
        predict,_=enc2(tx)
        l=loss(predict,ty)
        l.backward()
        optimizer2.step()
    print(l.data[0])


teX=Variable(torch.FloatTensor(n_enc1))
_,encoded2=enc2(teX)

encoded2=(encoded2.data).numpy()


dec=decoder()
optimizer3=optim.Adam(dec.parameters(),lr=0.001)

for i in range(epochs):
    for j in range(int(5000/batchsize)):
        np.random.shuffle(index)
        tx,ty=Variable(torch.FloatTensor(encoded2[j*batchsize:(j+1)*batchsize])),Variable(torch.FloatTensor(trainy[j*batchsize:(j+1)*batchsize]))
        optimizer3.zero_grad()
        predict=dec(tx)
        l=loss(predict,ty)
        l.backward()
        optimizer3.step()
    print(l.data[0])
            
        
enc1.eval()
teX=Variable(torch.FloatTensor(testX))
_,output1=enc1(teX)

output1=(output1.data)

o1=Variable(output1)

enc2.eval()
_,output2=enc2(o1)

output2=(output2.data)

o2,tey=Variable(output2),Variable(torch.FloatTensor(testy))
dec.eval()
output=dec(o2)



l=loss(output,tey)#got around 0.06
print(l.data[0])

output=(output.data).numpy()

#check
X=np.arange(0,20,0.1)    
plt.plot(X,output[50])
plt.xlabel('Angle [rad]')
plt.ylabel('sin(x)')
plt.axis('tight')
         