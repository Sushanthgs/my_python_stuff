# -*- coding: utf-8 -*-
"""
Created on Fri Jan  8 08:10:24 2021

@author: sushanthsgradlaptop2
"""

import numpy as np
from matplotlib import pyplot
from scipy import signal
I=pyplot.imread('C:\\Users\\sushanthsgradlaptop2\\Downloads\\cameraman.png')
I=I.astype(float)
I=(I-np.min(I))/(np.max(I)-np.min(I))
pyplot.imshow(I,cmap='gray')

def ng(I):
    Im=np.append(I[0,:].reshape(1,I.shape[1]),I[0:I.shape[0]-1,:],axis=0)-I
    return(Im)
def eg(I):
    Im=np.append(I[:,0].reshape(I.shape[0],1),I[:,0:I.shape[1]-1],axis=1)-I
    return(Im)
def sg(I):
    Im=np.append(I[1:I.shape[0],:],I[I.shape[0]-1,:].reshape(1,I.shape[1]),axis=0)-I
    return(Im)
def wg(I):
    Im=np.append(I[:,1:I.shape[1]],I[:,I.shape[1]-1].reshape(I.shape[0],1),axis=1)-I
    return(Im)



def get_w(Ip,k):
    return np.exp(-(Ip/k)**2)
#%
iter_v=200
m1=np.random.randn(I.shape[0],I.shape[1])
m1_r=m1.reshape(m1.shape[0]*m1.shape[1],1)
m1_r=(m1_r-np.mean(m1_r))/np.std(m1_r)
m1_r1=m1_r.reshape(I.shape[0],I.shape[1])
Ip_n=I+0.1*m1_r1
#
pyplot.imshow(Ip_n,cmap='gray',vmin=0,vmax=1)
#%
Ipm=Ip_n
k=0.1
st=0.1
reg_al=0.92
iter_num=20
#%
for i in range(iter_v):
    Ipm+=st*0.25*(get_w(ng(Ipm),k)*ng(Ipm)
               +get_w(sg(Ipm),k)*sg(Ipm)
               +get_w(eg(Ipm),k)*eg(Ipm)
               +get_w(wg(Ipm),k)*wg(Ipm))
    if(i%iter_num==0):
        Ipm=reg_al*Ipm+(1-reg_al)*signal.medfilt2d(Ipm)
#
#pyplot.imshow(Ip_n,cmap='gray',vmin=0,vmax=1) 
pyplot.figure()
pyplot.imshow(Ipm,cmap='gray',vmin=0,vmax=1)