import numpy as np
import matplotlib.pyplot as plt
from gan import gan as GAN
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('dim')
parser.add_argument('n')
parser.add_argument('eps')
parser.add_argument('exp_name')
args = parser.parse_args()
data_dim = int(args.dim)
data_size = int(args.n)
eps = float(args.eps)
eps_int = int(100 * eps)

exp_name = f'{args.exp_name}/dim{data_dim}n{data_size}eps{eps_int}'
os.makedirs(f'images_for_grad2/{exp_name}', exist_ok=True)


loss = []
for i in range(10):
    gan = GAN(data_dim=data_dim, eps=eps)
    gan.dist_init(setting='mu', true_mean=5, out_mean=0)
    gan.data_init(data_size=data_size, mc_ratio=3)
    gan.model_init()
    gan.optimizer_init(lr_d=0.2, lr_g=0.02, decay_par=0.4, reg_d=6e-5, reg_g=6e-5, update_D_iter=5)
    gan.fit(optim_iter=3000)
    loss.append(gan.score(100))
    # plt.subplots()
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(gan.l2_loss)
    plt.xlabel('optim step')
    plt.ylabel('l2 loss')
    plt.title(f'data dim is {gan.data_dim}')
    plt.subplot(1, 2, 2)
    plt.plot(np.array(gan.G_record))
    plt.xlabel('optim step')
    plt.ylabel('estimated value of mean')
    plt.title(f'data dim is {gan.data_dim}')
    plt.savefig(f'images_for_grad/{exp_name}-{i}.png')
loss = np.array(loss)

with open(f'images_for_grad2/{exp_name}.txt', 'w') as f:
    f.write(f'loss mean is {np.mean(loss)}')
    f.write(f'loss std  is {np.std(loss)}')
