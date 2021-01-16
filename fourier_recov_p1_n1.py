# -*- coding: utf-8 -*-
"""
Created on Fri Jan 15 21:58:09 2021

@author: sushanthsgradlaptop2
"""

import numpy as np
from matplotlib import pyplot 
from scipy import linalg
from scipy import fftpack
import time
I=pyplot.imread('C:\\Users\\sushanthsgradlaptop2\\Downloads\\peppers.png')
def sl(x,lam):
    return(np.sign(x)*(np.maximum(np.abs(x)-lam,0)))
msk=np.random.rand(I.shape[0],I.shape[1])>0.75
msk=msk.astype(int)
I_msk=I*msk
iter_v=500
lam=8000
lr=0.9
err=np.zeros([iter_v,1])
Y=0*I_msk
for i in range(iter_v):
    print(i)
    If=fftpack.fft2(Y)
    sm=sl(np.abs(If),lam*lr)*If/(1e-10+np.abs(If))
    Cr=fftpack.ifft2(sm)
    err[i]=np.sum(np.sum(msk*(np.real(Cr)-I_msk)**2))/np.sum(np.sum(msk))
    Y=Y+lr*msk*(I_msk-(Cr))
    
pyplot.subplot(2,2,1).imshow(I,cmap='gray')
pyplot.subplot(2,2,2).imshow(I_msk,cmap='gray')
pyplot.subplot(2,2,3).imshow(np.abs(Cr),cmap='gray')
pyplot.subplot(2,2,4).semilogy(np.abs(err[1:]-err[0:iter_v-1]))
