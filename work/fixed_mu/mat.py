import sys
sys.path.append("../")
import numpy as np
import numpy.linalg as LA
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')
from tqdm import tqdm
from numpy.lib.function_base import cov
from sys import argv
from libs.functions import sigmoid, g_lo, g_up, nearPD, sample_wise_vec_mat_vec, deriv_sigmoid, sample_wise_outer_product
from libs.create import create_out_cov, create_norm_data


n = int(argv[1])
m = 3*n
eps = float(argv[2])
data_dim = int(argv[3])
mu = float(argv[4])
mu_out = float(argv[5])
par_mu = np.full(data_dim, mu)
par_sd = np.identity(data_dim)
out_mu = np.full(data_dim, mu_out)
out_sd = np.identity(data_dim)
true_alpha = [par_mu, par_sd]

exper_iter = int(argv[6])
optim_iter = int(argv[7])
L = int(argv[8])
optim_method = str(argv[9])
dicay_par = float(argv[10])
par_reg1=float(argv[11])
learn_par = float(argv[12])
init_loc=float(argv[13])


# 分散を固定しない
out_cov = np.eye(data_dim) #create_out_cov(data_dim)
res_cov = [0 for _ in range(exper_iter)]
res_par = [0 for _ in range(exper_iter)]
for exp_index in (range(exper_iter)):
    data = create_norm_data(n, eps, par_mu, par_sd, out_mu, out_cov)
    cov_hist = []
    par_hist = []
    # 平均は次元ごとにロバスト、分散はロバストでない
    alpha = [np.full(data_dim, par_mu), np.cov(data, rowvar = False)]
    z = np.random.multivariate_normal(mean=alpha[0], cov=alpha[1], size = m)

    def discriminator(x, beta):
        size, data_dim = x.shape[0], x.shape[-1]
        beta = beta.reshape([data_dim,data_dim])
        value = sample_wise_vec_mat_vec(beta,x)
        return value

    size_par = data_dim**2
    par = np.random.normal(loc = 0, scale = 0.1, size = size_par)
    bias = np.array(np.mean( discriminator(z, par[0:size_par]) ))[np.newaxis]
    par = np.concatenate([par, bias], axis = 0) #(d**2 +1)
    cov_hist.append(alpha[1])
    par_hist.append(par)
    for j in tqdm(range(1, optim_iter+1)):
        z = np.random.multivariate_normal(mean=alpha[0], cov=alpha[1] , size = m)
        t0_z = discriminator(z, par[:-1]) - bias #shape (m,)
        t0_data = discriminator(data, par[:-1]) - bias # (n,)
        zzT = sample_wise_outer_product(z,z)
        xxT = sample_wise_outer_product(data, data)
        A=np.zeros([data_dim, data_dim, data_dim, data_dim])
        A_bias_col = np.zeros([data_dim,data_dim])
        A_b = np.zeros([data_dim,data_dim])
        A_bias_row = np.zeros([data_dim, data_dim])
        for i in range(data_dim):
            for k in range(data_dim):
                A[i][k] = -1/10*( np.mean(z[:,i][:,np.newaxis, np.newaxis]*z[:,k][:,np.newaxis,np.newaxis]*zzT, axis=0)\
                                + np.mean(data[:,i][:,np.newaxis, np.newaxis]*data[:,k][:,np.newaxis, np.newaxis]*xxT, axis = 0))             
            A_bias_col[i] = 1/10*np.mean(z[:,i][:,np.newaxis]*z, axis=0) +np.mean(data[:,i][:,np.newaxis]*data, axis=0)
            A_b[i] = -(np.mean(((deriv_sigmoid(t0_z)+t0_z/10)*z[:,i])[:,np.newaxis]*z, axis=0)\
                         -np.mean(((deriv_sigmoid(t0_data)-t0_data/10)*data[:,i])[:,np.newaxis]*data,axis=0)) 
            A_bias_row[i] = 1/10*(np.mean(z[:,i, np.newaxis]*z , axis=0)  +np.mean(data[:,i, np.newaxis]*data , axis=0))
        bias_bias = -0.2
        b_bias = ( np.mean(deriv_sigmoid(t0_z) + t0_z/10, axis=0) -np.mean(deriv_sigmoid(t0_data - t0_data/10),axis=0)) 


        A_bias_col = A_bias_col.reshape(data_dim**2, 1)    
        A_bias_row = A_bias_row.reshape(1,data_dim**2)    
        A_b = A_b.reshape(data_dim**2)

        A = A.T.reshape(data_dim,data_dim**2,data_dim).T.reshape(data_dim**2,data_dim**2)
        A = np.concatenate([A,A_bias_col], axis=1)
        A_bias_row = np.concatenate([A_bias_row, np.array(bias_bias)[np.newaxis, np.newaxis]], axis=1)
        A = np.concatenate([A,A_bias_row], axis=0)
        b = np.concatenate([A_b, b_bias[np.newaxis]], axis = 0)
        
        par = LA.lstsq(A,b)[0]

        alpha_m = alpha[0]; alpha_v = alpha[1]
        v_inv = np.linalg.inv(alpha_v)
        sigma_grad = (alpha_v - (z- alpha_m)[:,:,np.newaxis] * (z-alpha_m)[:, np.newaxis, :])/2
        sig_ = sigmoid(discriminator(z, par[:-1]) - par[-1])[:,np.newaxis]
        tmp_alpha_v = v_inv - learn_par/j**dicay_par * np.mean(sigma_grad*sig_[:,:,np.newaxis], axis = 0)
        alpha[1] = (LA.inv(tmp_alpha_v))

        cov_hist.append(alpha[1])
        par_hist.append(par)
    res_cov[exp_index] = cov_hist
    res_par[exp_index] = par_hist


import os
file_name=0
while True:
    path = "result/cov"+str(file_name)+".npy"
    if not os.path.isfile(path):
        break
    file_name+=1


path = "result/cov"+str(file_name)+".npy"
np.save(path,np.array(res_cov))
path = "result/par"+str(file_name)+".npy"
np.save(path,np.array(res_par))

cov = []; npcov=np.array(res_cov)

    
for i in range(exper_iter):
    loss_cov = LA.norm(npcov[i]-np.eye(data_dim), axis=(1,2))
    cov.append(loss_cov[-1])

average_loss_cov = str(np.mean(cov))[:6]


import datetime
date = str(datetime.datetime.now())[:-10][:10]
time = str(datetime.datetime.now())[:-10][10:]

with open("./exper_result.csv", mode="a") as f:
    f.write("\n")
    l =  [file_name, average_loss_cov ,n, eps, data_dim,
            mu, mu_out,
            par_reg1, learn_par, dicay_par,
            exper_iter, optim_iter, L, init_loc,date, time]
    l = list(map(str, l))
    f.writelines(" ,".join(l))