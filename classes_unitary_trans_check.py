# -*- coding: utf-8 -*-
"""
Created on Sat Dec 12 13:51:41 2020

@author: sushanthsgradlaptop2
"""
import numpy as np
from scipy import fftpack 
from matplotlib import pyplot
class dict_unitary:
    def __init__(self,dict_p,tr_dat,spp,iter_v):
        self.dict=dict_p
        self.coefs=np.zeros([dict_p.shape[0],tr_dat.shape[1]])
        self.spp=spp
        self.err=np.zeros([iter_v,1])
    def batch_thresh(self,spp,tr_dat,i):
        d_m=np.matmul(np.transpose(self.dict),tr_dat)
        ac=np.sort(np.abs(d_m),axis=0)[::-1]
        thr_val=ac[self.spp-1,:].reshape(1,ac.shape[1])
        self.coefs=d_m*(np.abs(d_m)>=thr_val)
        int_res=np.matmul(self.dict,self.coefs)-tr_dat
        self.err[i]=np.mean(np.mean(int_res**2))
        return(self)
    def update_dict(self,tr_dat):
        mat=np.matmul(self.coefs,np.transpose(tr_dat))
        [u,s,vh]=np.linalg.svd(mat)
        self.dict=np.matmul(np.transpose(vh),np.transpose(u))
        return(self)
    def train_dict(self,tr_dat,spp):
        for i in range(0,len(self.err)):
            self.batch_thresh(spp,tr_dat,i)
            self.update_dict(tr_dat)
#%%

#initialize generating transform matrix
class gen_unitary:
    def __init__(self,dat_dim,dat_size,spp):
       
        self.genmtx=init_gen_mat(dat_dim)
        self=init_coef_mat(self,dat_size,spp)
        self.dat=np.matmul(self.genmtx,self.coefs)
def init_gen_mat(dat_dim):
        ne=np.eye(dat_dim)
        genmtx=fftpack.fft(ne)
        genmtx=genmtx/np.sqrt(np.sum(np.abs(genmtx)**2,axis=0))
        am=np.real(genmtx)
        im=np.imag(genmtx)
        im=im[:,np.sum(np.abs(im),axis=0)!=0]
        genmtx=np.append(am,im,axis=1)
        genmtx/np.sqrt(np.sum(genmtx**2,axis=0))
        return(genmtx)
    
def init_coef_mat(self,dat_size,spp):
    np.random.seed(25000)

    coefs=np.zeros([self.genmtx.shape[1],dat_size])
    for i in range(dat_size):
            idx_val=(np.random.permutation(range(self.genmtx.shape[1]))).reshape(self.genmtx.shape[1],1)
            cm=idx_val[0:spp].ravel()
            coefs[cm,i]=10*np.random.randn(spp,1).ravel()
            
        #ravel returns flattened array can avoid problems with rank 1 arrays
    self.coefs=coefs
    
    return(self)
dat_size=1000
spp=5
dat_dim=256
g=gen_unitary(dat_dim,dat_size,spp)
#%%
ne2=np.eye(256)
dctmtx=fftpack.dct(ne2)
dctmtx=dctmtx/np.sqrt(np.sum(dctmtx**2,axis=0))
iter_v=100
d_c1=dict_unitary(dctmtx,g.dat,spp,iter_v)
d_c1.train_dict(g.dat,spp)