# -*- coding: utf-8 -*-
"""

@author: sushanthsgradlaptop2
"""

import numpy as np
    
import matplotlib
#define soft thresholding operator
def soft_t(x,lam):
    return(np.sign(x)*(np.maximum(np.abs(x)-lam*np.ones(x.shape),0)))
#initialize dictionary with random elements
def init_dict(dat_dim,num_atoms):
    dict_gt=np.random.rand(dat_dim,num_atoms)
    atom_ind=np.array(range(1,num_atoms))
    dict_gt[:,atom_ind]=dict_gt[:,atom_ind]-np.mean(dict_gt[:,atom_ind])
    cm=np.sqrt(np.sum(dict_gt[:,atom_ind]**2,axis=0))
    dict_gt[:,atom_ind]=dict_gt[:,atom_ind]/cm
    dict_gt[:,0]=np.ones([dat_dim,1]).ravel()/np.sqrt(dat_dim)
    return(dict_gt)

#initialize ground truth dictionary and coefficient

def gen_gt(dat_dim,dat_size,num_atoms,spp):
    dict_gt=init_dict(dat_dim,num_atoms)
    coefs=np.zeros([num_atoms,dat_size])
    for i in range(dat_size):
        idx_val=(np.random.permutation(range(num_atoms))).reshape(num_atoms,1)
        cm=idx_val[0:spp].ravel()
        coefs[cm,i]=10*np.random.randn(spp,1).ravel() 
        #ravel returns flattened array can avoid problems with rank 1 arrays
    dict_r={'dict_g': dict_gt,
            'coef': coefs}
    return(dict_r)
#perform ISTA iterations
def ISTA(dict_p,dat,lam,iter_v):
    d_co=np.matmul(np.transpose(d_m['dict_g']),d_m['dict_g'])
    v=np.linalg.eigvals(d_co)
    lr=1/np.max(np.abs(v))
    #proj=np.matmul(d_co,d_m['coef'])
    coefs=np.zeros([num_atoms,dat_size])
    cost_s=np.zeros([iter_v,1])
    for i in range(iter_v):
        G=np.matmul(np.transpose(dict_p),np.matmul(dict_p,coefs)-dat)
        cm_p=coefs-lr*(G)
        coefs=soft_t(cm_p,lam*lr)
        cost_s[i]=np.mean(np.mean((np.matmul(dict_p,coefs)-dat)**2))
        +np.mean(np.mean(np.abs(lam*lr*coefs)))
    dict_r={'dict_g':dict_p,'coef':coefs,'cost_v':cost_s}
    return(dict_r)

dat_size=1000
dat_dim=100
num_atoms=256
iter_v=1000
spp=5
d_m=gen_gt(dat_dim,dat_size,num_atoms,spp)   
    
#%% testing ISTA 
dat=np.matmul(d_m['dict_g'],d_m['coef'])
lam=0.01
mc=ISTA(d_m['dict_g'],dat,lam,iter_v)