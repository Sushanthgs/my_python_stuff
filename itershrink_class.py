# -*- coding: utf-8 -*-
"""
Created on Mon Jan 25 15:58:38 2021

@author: sushanthsgradlaptop2
"""
import numpy as np
from scipy import fftpack
from scipy import sparse
from matplotlib import pyplot as plt
import time
#initialize generating transform matrix
np.random.seed(25000)
ne=np.eye(392)
genmtx=fftpack.fft(ne)
genmtx=genmtx/np.sqrt(np.sum(np.abs(genmtx)**2,axis=0))
am=np.real(genmtx)
im=np.imag(genmtx)
im=im[:,np.sum(np.abs(im),axis=0)!=0]
genmtx=np.append(am,im,axis=1)
genmtx=genmtx/np.sqrt(np.sum(genmtx**2,axis=0))
def init_coef_mat(num_atoms,dat_size,spp):
    coefs=np.zeros([num_atoms,dat_size])
    for i in range(dat_size):
        idx_val=(np.random.permutation(range(num_atoms))).reshape(num_atoms,1)
        cm=idx_val[0:spp].ravel()
        coefs[cm,i]=10*np.random.randn(spp,1).ravel()
        #ravel returns flattened array can avoid problems with rank 1 arrays
    return(coefs)
#
init_coefs=init_coef_mat(genmtx.shape[1],6000,5)
init_coefs_sp=sparse.coo_matrix(init_coefs)
dat_init=genmtx @ init_coefs_sp
#%%
class iter_shrink:
    def __init__(self, mat,dat,iter_v,lam):
        self.D=mat
        self.sp_coef=np.zeros([mat.shape[1],dat.shape[1]])
        self.dat=dat
        self.iter_v=iter_v
        self.lam=lam
        self.err=np.zeros([iter_v,1])
        self.D_co=np.transpose(self.D) @ self.D
        self.lr=1/np.max(np.abs(np.linalg.eigvals(self.D_co)))
        self.coef_v=np.transpose(self.D) @ dat
    def __get_grad__(self):
        return self.D_co @ self.sp_coef-self.coef_v
    def soft_t(self):
        gm=self.sp_coef-self.lr*self.__get_grad__()
        return np.sign(gm)*(np.maximum(np.abs(gm)-self.lam*self.lr,0))
    def __next__(self):
        self.sp_coef=self.soft_t()
    def __iter__(self):
        for i in range(self.iter_v):
            print(i)
            self.__next__()
            self.err[i]=np.mean(np.mean(self.__get_grad__()**2))
#%%
         
iter_test=iter_shrink(genmtx,dat_init,100,7)
iter_test.__iter__()
#%

    