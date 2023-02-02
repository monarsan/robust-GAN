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
from libs.functions import sigmoid, g_lo, g_up, nearPD, sample_wise_vec_mat_vec,\
deriv_sigmoid, sample_wise_outer_product, init_covariance, init_discriminator
from libs.create import create_sparse_cov, create_norm_data
import matplotlib.pyplot as plt
from libs.utils import slack_massage, tqdm_slack
import seaborn as sns


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
exper_name = str(argv[14])
cov0 = str(argv[15])
u0 = str(argv[16])

import os
file_name=0
while True:
    path = "result/cov"+str(file_name)+".npy"
    if not os.path.isfile(path):
        break
    file_name+=1

res_cov = [0 for _ in range(exper_iter)]
res_par = [0 for _ in range(exper_iter)]
res_true_cov=[]
res_out_cov = []
def discriminator(x, beta):
    size, data_dim = x.shape[0], x.shape[-1]
    beta = beta.reshape([data_dim,data_dim])
    value = sample_wise_vec_mat_vec(beta,x)
    return value

for exp_index in tqdm_slack(exper_iter):
    slack_massage("%d/%d"%(exp_index+1,exper_iter))
    par_sd = create_sparse_cov(data_dim)
    out_cov = create_sparse_cov(data_dim)
    res_true_cov.append(par_sd); res_out_cov.append(out_cov)
    data = create_norm_data(n, eps, par_mu, par_sd, out_mu, out_cov)
    cov_hist = []
    par_hist = []


    #wandb init
    config = dict(
    learning_rate = learn_par,
    dicay_parameter = dicay_par,
    outlier_mu = out_mu,
    covariance_setting = "sparse"
    )   

    import wandb
    wandb.init(
        project="robust scatter estimation",
        entity="robust-gan",
        config=config
    )
    log_name = str(file_name)+'-'+exper_name+'-'+str(exp_index)
    wandb.run.name = log_name
    wandb.run.save()
    dict_wandb=dict()


    B = init_covariance(data, cov0)
    alpha = [np.full(data_dim, par_mu), B]
    z = np.random.multivariate_normal(mean=alpha[0], cov=B@B.T, size = m)

    size_par = data_dim**2
    par = init_discriminator(data, u0)
    #par = np.random.normal(loc = 0, scale = 0.1, size = size_par)
    bias = np.array(np.mean(discriminator(z, par[0:size_par]) ))[np.newaxis]
    par = np.concatenate([par, bias], axis = 0) #(d**2 +1)
    cov_hist.append(alpha[1]@alpha[1].T)
    par_hist.append(par)
    for j in range(optim_iter):
        z = np.random.multivariate_normal(mean=alpha[0], cov=alpha[1]@alpha[1].T , size = m)
        zzT = sample_wise_outer_product(z,z)
        xxT = sample_wise_outer_product(data, data)
        A=np.zeros([data_dim, data_dim, data_dim, data_dim])
        A_bias_col = np.zeros([data_dim,data_dim])
        A_b = np.zeros([data_dim,data_dim])
        A_bias_row = np.zeros([data_dim, data_dim])
        l=0
        while (l<L) :
            t0_z = discriminator(z, par[:-1]) - par[-1] #shape (m,)
            t0_data = discriminator(data, par[:-1]) - par[-1] # (n,)
            for i in range(data_dim):
                for k in range(data_dim):
                    A[i][k] = -1/10*( np.mean(z[:,i][:,np.newaxis, np.newaxis]*z[:,k][:,np.newaxis,np.newaxis]*zzT, axis=0)\
                                    + np.mean(data[:,i][:,np.newaxis, np.newaxis]*data[:,k][:,np.newaxis, np.newaxis]*xxT, axis = 0))
                # reguralization for A
                A[i][i] -= np.eye(data_dim)/(n**2)
                A_bias_col[i] = 1/10*(np.mean(z[:,i][:,np.newaxis]*z, axis=0) +np.mean(data[:,i][:,np.newaxis]*data, axis=0))
                A_b[i] = -(np.mean(((deriv_sigmoid(t0_z)+t0_z/10)*z[:,i])[:,np.newaxis]*z, axis=0)\
                            -np.mean(((deriv_sigmoid(t0_data)-t0_data/10)*data[:,i])[:,np.newaxis]*data,axis=0)) 
                A_bias_row[i] = 1/10*(np.mean(z[:,i, np.newaxis]*z , axis=0)  +np.mean(data[:,i, np.newaxis]*data , axis=0))
            bias_bias = -0.2
            b_bias = ( np.mean(deriv_sigmoid(t0_z) + t0_z/10, axis=0) -np.mean(deriv_sigmoid(t0_data - t0_data/10),axis=0)) 

            A_bias_col_reshaped = A_bias_col.reshape(data_dim**2, 1)    
            A_bias_row_reshaped = A_bias_row.reshape(1,data_dim**2)    
            A_b_reshaped = A_b.reshape(data_dim**2)

            A_reshaped = A.T.reshape(data_dim,data_dim**2,data_dim).T.reshape(data_dim**2,data_dim**2)
            A_concated = np.concatenate([A_reshaped, A_bias_col_reshaped], axis=1)
            A_bias_row_concated = np.concatenate([A_bias_row_reshaped, np.array(bias_bias)[np.newaxis, np.newaxis]], axis=1)
            A_ = np.concatenate([A_concated,A_bias_row_concated], axis=0)
            b = np.concatenate([A_b_reshaped, b_bias[np.newaxis]], axis = 0)
            lr = learn_par/(j+1)**dicay_par
            par =  par*(1-lr) + lr*LA.solve(A_,b)
            l+= 1

        z = np.random.multivariate_normal(mean=alpha[0], cov=np.eye(data_dim) , size = m)
        alpha_m = alpha[0]; alpha_v = alpha[1]
        ABZ = (z@par[:-1].reshape(data_dim, data_dim))@(alpha_v+alpha_v.T)
        sigma_grad = sample_wise_outer_product(ABZ, z) #(m, d, d)
        sig_ = deriv_sigmoid(discriminator(z@alpha_v, par[:-1]) - par[-1])[:,np.newaxis] #(m,)
        tmp_alpha_v = alpha_v - learn_par/(j+1)**dicay_par * np.mean(sigma_grad*sig_[:,:,np.newaxis], axis = 0)
        alpha[1] = tmp_alpha_v

        cov_hist.append(alpha[1]@alpha[1].T)
        par_hist.append(par)
    res_cov[exp_index] = cov_hist
    res_par[exp_index] = par_hist
    

    #to numpy
    tmp_cov = np.array(cov_hist)
    par=np.array(par_hist)
    #wandb log
    loss_cov = LA.norm(tmp_cov-par_sd, ord=2, axis=(1,2))
    loss_data = np.concatenate([np.arange(len(loss_cov))[:,np.newaxis], loss_cov[:,np.newaxis]], axis=1).tolist()
    table = wandb.Table(data=loss_data, columns=['step','op norm loss'])
    line_plot = wandb.plot.line(table, x='step', y='op norm loss', title='op norm loss')
    dict_wandb['op norm loss'] = line_plot

    loss_cov = LA.norm(tmp_cov-par_sd, ord='fro', axis=(1,2))
    loss_data = np.concatenate([np.arange(len(loss_cov))[:,np.newaxis], loss_cov[:,np.newaxis]], axis=1).tolist()
    table = wandb.Table(data=loss_data, columns=['step','fro norm loss'])
    line_plot = wandb.plot.line(table, x='step', y='fro norm loss', title='fro norm loss')
    dict_wandb['fro norm loss'] = line_plot

    bias_log = par[:,-1]
    bias_data = np.concatenate([np.arange(len(bias_log))[:,np.newaxis], bias_log[:,np.newaxis]], axis=1).tolist()
    table = wandb.Table(data=bias_data, columns=['step','bias'])
    line_plot = wandb.plot.line(table, x='step', y='bias', title='bias')
    dict_wandb['bias'] = line_plot

    A_log = LA.norm(par[:,:-1].reshape(par.shape[0],data_dim,data_dim), ord="fro", axis=(1,2))
    A_data = np.concatenate([np.arange(len(A_log))[:,np.newaxis], A_log[:,np.newaxis]], axis=1).tolist()
    table = wandb.Table(data=A_data, columns=['step','fro norm of A'])
    line_plot = wandb.plot.line(table, x='step', y='fro norm of A', title='fro norm of A')
    dict_wandb['fro norm of A'] = line_plot

    data12= data[:,:2]
    table = wandb.Table(data=data12, columns = ["dim1", "dim2"])
    dict_wandb["data dim1 and dim2"] = wandb.plot.scatter(table, 'dim1', 'dim2', 'data dim1 and dim2')

    data12= data[:,2:4]
    table = wandb.Table(data=data12, columns = ["dim3", "dim4"])
    dict_wandb["data dim3 and dim4"] = wandb.plot.scatter(table, 'dim3', 'dim4', 'data dim3 and dim4')

    simple_labels = list(map(str,list(range(data_dim))))
    dict_wandb['true cov'] = wandb.plots.HeatMap(simple_labels,simple_labels, par_sd, show_text = True)

    dict_wandb['out cov'] = wandb.plots.HeatMap(simple_labels,simple_labels, out_cov, show_text = True)
    dict_wandb['est cov'] = wandb.plots.HeatMap(simple_labels,simple_labels, tmp_cov[-1], show_text = True)
    dict_wandb['mle'] = wandb.plots.HeatMap(simple_labels,simple_labels, init_discriminator(data, 'mle').reshape(data_dim,data_dim), show_text = True)
    dict_wandb['kendall'] = wandb.plots.HeatMap(simple_labels,simple_labels, init_discriminator(data, 'kendall').reshape(data_dim,data_dim), show_text = True)
    dict_wandb['final A'] = wandb.plots.HeatMap(simple_labels,simple_labels, par[-1,:-1].reshape(data_dim, data_dim), show_text = True)

    wandb.log(dict_wandb)
    wandb.finish()

path = "result/cov"+str(file_name)+".npy"
np.save(path,np.array(res_cov))
path = "result/par"+str(file_name)+".npy"
np.save(path,np.array(res_par))
path = "result/true_par"+str(file_name)+".npy"
np.save(path,np.array(res_true_cov))
path = "result/out_cov"+str(file_name)+".npy"
np.save(path,np.array(res_out_cov))


cov = []; npcov=np.array(res_cov)
loss_plot = []
    
for i in range(exper_iter):
    loss_cov = LA.norm(npcov[i]-res_true_cov[i], axis=(1,2))
    loss_plot.append(loss_cov)
    cov.append(loss_cov[-1])

average_loss_cov = str(np.mean(cov))[:6]
std_loss_cov = str(np.std(cov))[:6]


import datetime
date = str(datetime.datetime.now())[:-10][:10]
time = str(datetime.datetime.now())[:-10][10:]



with open("./exper_result.csv", mode="a") as f:
    f.write("\n")
    l =  [file_name, average_loss_cov ,n, eps, data_dim,
            mu, mu_out,
            par_reg1, learn_par, dicay_par,
            exper_iter, optim_iter, L, init_loc,date, time, std_loss_cov]
    l = list(map(str, l))
    f.writelines(" ,".join(l))