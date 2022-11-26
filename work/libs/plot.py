import numpy as np
import numpy.linalg as LA
import matplotlib.pyplot as plt
import sys
sys.path.append("../")
from libs.functions import sigmoid

def plot_mu_cov(mean, res_mean, res_cov, res_par):
    mean = np.array(res_mean)
    cov = np.array(res_cov)
    par = np.array(res_par)

    exper_iter, optim_iter, data_dim = mean.shape

    mean_vec = np.full(data_dim, mean)
    cov_mat = np.eye(data_dim)
    
    mean_end = []; cov_end = []
    for i in range(exper_iter):
        loss_mean = LA.norm(mean[i] - mean_vec, ord = 2, axis = 1)
        plt.plot(loss_mean)
        mean_end.append(loss_mean[-1])
    print("mean ave : %.5f" %np.mean(mean_end))
    print("mean std : %.5f" %np.std(mean_end))


    plt.subplots()
    for i in range(exper_iter):
        loss_cov = LA.norm(cov[i] - cov_mat, ord = 2, axis = (1,2))
        plt.plot(loss_cov)
        cov_end.append(loss_cov[-1])
    print("cov  ave : %.5f" %np.mean(cov_end))
    print("cov  std : %.5f" %np.std(cov_end))

    for i in range(exper_iter):
        plt.subplots()
        for j in range(par.shape[2]):
            plt.plot(par[i,:,j], label=str(j))
            plt.legend()
        


def plot_sig(res_mu, res_cov, res_par, discriminator):
    mean = np.array(res_mu)
    cov = np.array(res_cov)
    par = np.array(res_par)
    exper_iter, optim_iter, data_dim = mean.shape

    for j in range(exper_iter):
        l = []
        for i in range(optim_iter):
            z = np.random.multivariate_normal(mean[j,i], cov[j,i], size = 3000)
            par_ = par[j,i]
            sig_ = sigmoid(np.dot(discriminator(z, par_[:-1])- par_[2*data_dim])[:,np.newaxis]).mean()
            l.append(sig_)
        plt.plot(l)



def plot_loss(res, res_par, par_mu=0, ylim = False):
    mu_result = np.array(res)
    nppar = np.array(res_par)
    num_graph_row = 3
    exper_iter = mu_result.shape[0]
    optim_iter = mu_result.shape[1]
    plt.figure(figsize=(4*num_graph_row,4*exper_iter))
    half = int(optim_iter/2)
    mean=[]
    for i in (range(exper_iter)):

        #loss of u norm
        plt.subplot(exper_iter, num_graph_row, num_graph_row*i +1)
        plt.title("l2 norm of u")
        if ylim:
            plt.ylim(0.2,0.8)
        loss = np.linalg.norm(nppar[i,:, :-1], ord = 2, axis=1)
        plt.plot(range(len(loss)), loss)   

        #bias
        plt.subplot(exper_iter, num_graph_row, num_graph_row*i +2)
        plt.title("bias")
        plt.plot(range(len(nppar[i,:,-1])), nppar[i,:, -1])       

        # loss of mu
        plt.subplot(exper_iter, num_graph_row, num_graph_row*i +3)
        plt.title("l2 loss")
        if ylim:
            plt.ylim(0,0.2)
        loss = np.linalg.norm(mu_result[i]-par_mu, ord = 2, axis=1)
        mean.append(loss[-1])
        plt.plot(range(len(loss)), loss)
    print("average mean is : %.4f"%np.mean(mean))



def plot_2dim(res, res_par, par_mu=0,ylim = True):
    mu_result = np.array(res)
    nppar = np.array(res_par)
    num_graph_row = 6
    exper_iter = mu_result.shape[0]
    optim_iter = mu_result.shape[1]
    plt.figure(figsize=(4*num_graph_row,4*exper_iter))
    half = int(optim_iter/2)
    mean=[]
    for i in (range(exper_iter)):
        # mu
        plt.subplot(exper_iter,num_graph_row, num_graph_row*i +1)
        plt.title("estimaterd mu")
        # plt.ylim(4.75, 5.3)
        # plt.xlim(4.75, 5.3)
        plt.scatter(mu_result[i,:half,0], mu_result[i,:half,1], color = "m")
        plt.scatter(mu_result[i,half:,0], mu_result[i,half:,1], color = "c")
        plt.scatter(mu_result[i,0,0], mu_result[i,0,1], c="red")
        plt.scatter(mu_result[i,-1,0], mu_result[i,-1,1], c="blue")
        #plt.scatter(5, 5, color = "black")

        # par1 2
        plt.subplot(exper_iter, num_graph_row, num_graph_row*i +2)
        plt.title("parameter of u")
        plt.scatter(nppar[i,half:,0], nppar[i,half:,1], color = "c")
        plt.scatter(nppar[i,:half,0], nppar[i,:half,1], color = "m")
        plt.scatter(nppar[i,0,0], nppar[i,0,1], c="red")
        plt.scatter(nppar[i,-1,0], nppar[i,-1,1], c="blue")
        
        # par3 4
        plt.subplot(exper_iter, num_graph_row, num_graph_row*i +3)
        plt.title("parameter of u")
        plt.scatter(nppar[i,half:,2], nppar[i,half:,3], color = "c")
        plt.scatter(nppar[i,:half,2], nppar[i,:half,3], color = "m")
        plt.scatter(nppar[i,0,2], nppar[i,0,3], c="red")
        plt.scatter(nppar[i,-1,2], nppar[i,-1,3], c="blue")

        #loss of u norm
        plt.subplot(exper_iter, num_graph_row, num_graph_row*i +4)
        plt.title("l2 norm of u")
        if ylim:
            plt.ylim(0.2,0.8)
        loss = np.linalg.norm(nppar[i,:, :-1], ord = 2, axis=1)
        plt.plot(range(len(loss)), loss)   

        #bias
        plt.subplot(exper_iter, num_graph_row, num_graph_row*i +5)
        plt.title("bias")
        plt.plot(range(len(nppar[i,:,-1])), nppar[i,:, -1])       

        # loss of mu
        plt.subplot(exper_iter, num_graph_row, num_graph_row*i +6)
        plt.title("l2 loss")
        if ylim:
            plt.ylim(0,0.2)
        loss = np.linalg.norm(mu_result[i]-par_mu, ord = 2, axis=1)
        mean.append(loss[-1])
        plt.plot(range(len(loss)), loss)
    print("average mean is : %.4f"%np.mean(mean))