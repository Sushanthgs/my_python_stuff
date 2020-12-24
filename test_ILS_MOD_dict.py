# -*- coding: utf-8 -*-
"""
Created on Wed Dec 23 13:31:26 2020

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
#
    
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




def dict_update(coef_mat,tr_dat):
    d_up=np.matmul(tr_dat,np.linalg.pinv(coef_mat))
    return(d_up)

def sp_code(genmtx,tr_dat,spp):
    z=np.zeros([genmtx.shape[1],dat_init.shape[1]])
    for i in range(dat_init.shape[1]):
        d_i=dat_init[:,i]
        g=OMP_iter(genmtx,d_i,spp)
        z[g['locs'],i]=g['vals'].ravel()
    return(z)

def train_dict_ILS_MOD(dict_p,tr_dat,spp,iter_v,mean_flag):
    err=np.zeros([iter_v,1])
    dict_train={}
    for i in range(iter_v):
        print(i)
        if(i==0 and mean_flag==1):
            coef_m=sp_code(dict_p,tr_dat,spp)
        else:
            if(mean_flag==1):
                d_c=tr_dat-np.matmul(dict_p[:,0][:,None], coef_m[0,:][None,:])
                coef_temp=sp_code(dict_p[:,1:],d_c,spp)
                coef_m=np.append(coef_m[0,:][None,:],coef_temp,axis=0)
            else:
                coef_m=sp_code(dict_p,tr_dat,spp)
        res=np.matmul(dict_p,coef_m)-tr_dat
        err[i]=np.mean(np.mean(res**2))
        print(err[i])          
        if(mean_flag==1):
             d_c=tr_dat-np.matmul(dict_p[:,0][:,None], coef_m[0,:][None,:])
             d_up=dict_update(coef_m[1:,:],d_c)
             d_up=np.append(np.ones([dict_p.shape[0],1]),d_up-np.mean(d_up),axis=1)
             d_up=d_up/(np.sqrt(np.sum(d_up**2,axis=0)))
        else:
            d_up=dict_update(coef_m,tr_dat)
            d_up=d_up/(np.sqrt(np.sum(d_up**2,axis=0)))
        dict_p=d_up
       
    dict_train['dict']=d_up
    dict_train['err']=err
    return(dict_train)
#%%
n_atoms=400
spp=5
iter_v=40

init_coefs=init_coef_mat(genmtx.shape[1],10000,5)
dat_init=np.matmul(genmtx,init_coefs)
dict_init=np.random.rand(genmtx.shape[0],n_atoms-1)
dict_init=dict_init-np.mean(dict_init)
dict_init=np.append(np.ones([dict_init.shape[0],1]),dict_init,axis=1)
dict_init=dict_init/np.sqrt(np.sum(dict_init**2,axis=0))
#%%
d_train=train_dict_ILS_MOD(dict_init,dat_init,spp,iter_v,1)
code_sp_learn_dict=sp_code(d_train['dict'],dat_init,spp)
rec_err=np.matmul(d_train['dict'],code_sp_learn_dict)-dat_init
#%%
ind_val=69
pyplot.subplot(2,2,1).imshow(dict_init,vmin=-0.01,vmax=0.01)
pyplot.title('Initial dictionary (Random)')
pyplot.axis('off')
pyplot.subplot(2,2,2).imshow(d_train['dict'],vmin=-0.01,vmax=0.01)
pyplot.title('Learned dictionary (Trained)')
pyplot.axis('off')
pyplot.subplot(2,2,3).plot(dict_init[:,ind_val])
pyplot.title('Atom: '+str(ind_val)+' of initial dictionary')
pyplot.subplot(2,2,4).plot(d_train['dict'][:,ind_val])
pyplot.title('Atom: '+str(ind_val)+' of trained dictionary')
