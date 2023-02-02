from gan import gan as GAN
import matplotlib.pyplot as plt
import numpy as np

def plot_eps(times = 10):
    eps_list = [0.05, 0.1, 0.15, 0.2]
    mean = np.zeros(len(eps_list))
    sigma = np.zeros(len(eps_list))
    for j, eps in enumerate(eps_list):
        tmp_result = np.zeros(times)
        plt.subplots()
        for i in range(times):
            gan = GAN(10, eps)
            gan.dist_init('sigma', 0, 5)
            gan.data_init(1000)
            gan.model_init()
            gan.optimizer_init(0.07, 0.1, 0.12, 0, 0, 1, 1)
            gan.fit(100)
            tmp_result[i] = gan.l2_loss[-1]
            plt.plot(gan.l2_loss)
        mean[j] = np.mean(tmp_result)
        sigma[j] = np.std(tmp_result)
    return  np.array(eps_list), mean, sigma


def plot_n(times = 20):
    n_list = [100, 300, 500, 750, 1000]
    mean = np.zeros(len(n_list))
    sigma = np.zeros(len(n_list))
    for j, n in enumerate(n_list):
        tmp_result = np.zeros(times)
        plt.subplots()
        for i in range(times):
            gan = GAN(10, 0.2)
            gan.dist_init('sigma', 0, 5)
            gan.data_init(n)
            gan.model_init()
            gan.optimizer_init(0.07, 0.1, 0.12, 0, 0, 1, 1)
            gan.fit(200)
            tmp_result[i] = gan.l2_loss[-1]
            plt.plot(gan.l2_loss)
        mean[j] = np.mean(tmp_result)
        sigma[j] = np.std(tmp_result)
    return np.array(n_list), mean, sigma

