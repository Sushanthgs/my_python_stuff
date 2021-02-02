# -*- coding: utf-8 -*-
"""
Created on Mon Feb  1 23:20:59 2021
testing backtracking line search
@author: sushanthsgradlaptop2
"""

import numpy as np
import matplotlib.pyplot as plt
dat_rand=np.random.randn(100,2)
dat_n1_s=dat_rand @ np.array([2.4,3.6]).reshape(2,1)
coef_init=np.zeros([2,1])
iter_v=100
err=np.zeros([iter_v,1])
lr_init=1
def funcval(mat,c,targ):
    return np.sum((mat@c-targ)**2)

#backtracking function
def backtrack_ls(dat_init,coef_init_p,targ,a,b,l_r):
    grad=np.transpose(dat_rand)@(dat_rand@coef_init_p-dat_n1_s)
    coef_init_n=coef_init_p-l_r*grad
    f1=funcval(dat_init,coef_init_n,targ)
    f2=funcval(dat_init,coef_init_p,targ)-a*l_r*np.sum(grad**2)
    while(f1>f2):
        l_r=l_r*b
        coef_init_n=coef_init_p-l_r*grad
        f1=funcval(dat_init,coef_init_n,targ)
        f2=funcval(dat_init,coef_init_p,targ)-a*l_r*np.sum(grad**2)
    
    return l_r


        
    
    
lr_v=lr_init
for i in range(iter_v):
    if(i%20==0): #added to see where the learning rate changes (inflection on graph)
        lr_n=backtrack_ls(dat_rand,coef_init,dat_n1_s,0.5,0.99,lr_v)

        
    grad=np.transpose(dat_rand)@(dat_rand@coef_init-dat_n1_s)
    coef_init-=lr_n*grad
    lr_v=lr_n
    err[i]=np.sum((dat_rand@coef_init-dat_n1_s)**2)
plt.semilogy(err)