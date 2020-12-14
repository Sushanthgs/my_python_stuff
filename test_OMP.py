# -*- coding: utf-8 -*-
"""
Created on Mon Dec 14 10:54:44 2020

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

#initialize generating coefficient matrix
def init_coef_mat(num_atoms,dat_size,spp):
    coefs=np.zeros([num_atoms,dat_size])
    for i in range(dat_size):
        idx_val=(np.random.permutation(range(num_atoms))).reshape(num_atoms,1)
        cm=idx_val[0:spp].ravel()
        coefs[cm,i]=10*np.random.randn(spp,1).ravel()
        #ravel returns flattened array can avoid problems with rank 1 arrays
    return(coefs)
#
init_coefs=init_coef_mat(genmtx.shape[1],1000,5)
dat_init=np.matmul(genmtx,init_coefs)
#%%
    
def OMP_iter(genmtx, d_i, spp):
    x = d_i
    loc = []
    res={}
    for i in range(spp):
        am = np.matmul(np.transpose(genmtx), d_i).reshape([genmtx.shape[1], 1])
        l1 = np.argmax(np.abs(am))
        
        loc.append(l1)
        act_set = genmtx[:, loc]
        a = np.matmul(np.linalg.pinv(act_set), x)
        d_i = x-np.matmul(act_set, a)
        if(np.sum(d_i**2) < 1e-6):
            break
    res['vals']=a.reshape([a.shape[0],1])
    res['locs']=loc
    return(res)
spp=5
z=np.zeros([genmtx.shape[1],dat_init.shape[1]])
for i in range(dat_init.shape[1]):
    d_i=dat_init[:,i]
    g=OMP_iter(genmtx,d_i,spp)
    z[g['locs'],i]=g['vals'].ravel()

res=np.matmul(genmtx,z)-dat_init
recon_err=np.mean(np.mean(res**2))
print(recon_err)