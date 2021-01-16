# -*- coding: utf-8 -*-
"""
Created on Fri Jan 15 17:00:14 2021

@author: sushanthsgradlaptop2

my implementation of compressed sensing recovery using nuclear norm minimization 
' A Singular Value Thresholding Algorithm for Matrix Completion' Cai, Candes, Shen
SDP, convex relaxation of rank minimization problem
"""

import numpy as np
from matplotlib import pyplot 
import time
I=pyplot.imread('C:\\Users\\sushanthsgradlaptop2\\Downloads\\cameraman.png')
def sl(x,lam):
    return(np.sign(x)*(np.maximum(np.abs(x)-lam,0)))
msk=np.random.rand(I.shape[0],I.shape[1])>0.5
I_msk=I*msk

def rec(x,msk,iter_v,lam,lr):
    Y=0*x
    err=np.zeros([iter_v,1])
    rs={}
    for i in range(iter_v):
        [U,S,V]=np.linalg.svd(Y)
        Cs=sl(S,lam*lr)
        Cr=U @ np.diag(Cs) @ (V)       
        res=-(Cr)+x
        Y=Y+lr*msk*(res)
        err[i]=np.sum(np.sum((msk*res)**2))/np.sum(np.sum(msk))
    rs['recon']=(Cr)
    rs['err']=err
    return(rs)

res_m=rec(I_msk,msk.astype(int),1000,100,1)
pyplot.subplot(2,2,1).imshow(I,cmap='gray',vmin=0,vmax=1)
pyplot.subplot(2,2,2).imshow(I_msk,cmap='gray',vmin=0,vmax=1)
pyplot.subplot(2,2,3).imshow(np.abs(res_m['recon']),cmap='gray',vmin=0,vmax=1)
pyplot.subplot(2,2,4).semilogy(res_m['err'])
pyplot.grid('on')
        
    
    