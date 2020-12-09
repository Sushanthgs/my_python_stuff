# -*- coding: utf-8 -*-
"""
Created on Wed Dec  9 13:46:14 2020

@author: sushanthsgradlaptop2
"""

#transform learning function Ravishankar and Bresler 
#IEEE Transactions on signal processing 2015

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
#%% define sparse coding function

def batch_thresh(dict_s,dat,spp):
    d_m=np.matmul(np.transpose(dict_s),dat)
    ac=np.sort(np.abs(d_m),axis=0)[::-1]
    thr_val=ac[spp-1,:].reshape(1,ac.shape[1])
    coefs=d_m*(np.abs(d_m)>=thr_val)
    return(coefs)
#%% train transform
def train_dict_unitary(dict_p,dat,spp,iter_v,l2,l3):
    err=np.zeros([iter_v,1])
    dat_co=np.matmul(dat,np.transpose(dat))
    for i in range(iter_v):
        b=batch_thresh(dict_p,dat,spp)
        e_v=(np.matmul(dict_p,b)-dat)**2
        err[i]=np.mean(np.mean(e_v,axis=0))
        mat=dat_co+l3*np.eye(dat_co.shape[0])
        [u,s,vh]=np.linalg.svd(mat)
        LL=np.matmul(u,np.matmul(s**0.5,vh))
        #svd returns vh (already transposed)

        Lpi=np.linalg.pinv(LL)
        sdiag=np.diag(s).reshape([s.shape[0],1])
        g_val=0.5*(sdiag+np.sqrt(sdiag**2+2*l2))
        Ip1=np.matmul(np.transpose(vh),np.diag(g_val))
        Ip2=np.matmul(Ip1,np.transpose(u))
        dict_p=np.matmul(Ip2,Lpi)
        dict_p=dict_p/(np.sqrt(np.sum(dict_p**2,axis=0)))
    train_dict={'dict_train':dict_p,
                'coefs':b,
                'err':err}
    return(train_dict) 