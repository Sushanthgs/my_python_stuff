# -*- coding: utf-8 -*-
"""
Created on Tue Dec  8 19:45:38 2020

@author: sushanthsgradlaptop2
"""

import numpy as np
from scipy import fftpack 
from matplotlib import pyplot
#initialize generating transform matrix
np.random.seed(25000)
ne=np.eye(256)
genmtx=fftpack.fft(ne)
genmtx=genmtx/np.sqrt(np.sum(np.abs(genmtx)**2,axis=0))
am=np.real(genmtx)
im=np.imag(genmtx)
im=im[:,np.sum(np.abs(im),axis=0)!=0]
genmtx=np.append(am,im,axis=1)
genmtx=genmtx/np.sqrt(np.sum(genmtx**2,axis=0))
#%%
#initialize generating coefficient matrix
def init_coef_mat(num_atoms,dat_size,spp):
    coefs=np.zeros([num_atoms,dat_size])
    for i in range(dat_size):
        idx_val=(np.random.permutation(range(num_atoms))).reshape(num_atoms,1)
        cm=idx_val[0:spp].ravel()
        coefs[cm,i]=10*np.random.randn(spp,1).ravel()
        #ravel returns flattened array can avoid problems with rank 1 arrays
    return(coefs)
#%% 
#initialize transform (Unitary)

def batch_thresh(dict_s,dat,spp):
    d_m=np.matmul(np.transpose(dict_s),dat)
    ac=np.sort(np.abs(d_m),axis=0)[::-1]
    thr_val=ac[spp-1,:].reshape(1,ac.shape[1])
    coefs=d_m*(np.abs(d_m)>=thr_val)
    return(coefs)

def train_dict_unitary(dict_p,dat,spp,iter_v):
    err=np.zeros([iter_v,1])
    for i in range(iter_v):
        b=batch_thresh(dict_p,dat,spp)
        e_v=(np.matmul(dict_p,b)-dat)**2
        err[i]=np.mean(np.mean(e_v,axis=0))
        mat=np.matmul(b,np.transpose(dat))
        [u,s,vh]=np.linalg.svd(mat)
        #svd returns vh (already transposed)
        dict_p=np.matmul(np.transpose(vh),np.transpose(u))
    train_dict={'dict_train':dict_p,
                'coefs':b,
                'err':err}
    return(train_dict)    

dat_size=1000
spp=5
coef_vals=init_coef_mat(genmtx.shape[1],dat_size,spp)
dat=np.matmul(genmtx,coef_vals)
ne2=np.eye(256)
dctmtx=fftpack.dct(ne2)
dctmtx=dctmtx/np.sqrt(np.sum(dctmtx**2,axis=0))
iter_v=100

train_dict=train_dict_unitary(dctmtx,dat,spp,iter_v)
  
    
    