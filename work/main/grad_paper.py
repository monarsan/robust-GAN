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
dir = 'images_for_grad_mean0_out5'

exp_name = f'{args.exp_name}/dim{data_dim}n{data_size}eps{eps_int}'
os.makedirs(f'{dir}/{exp_name}', exist_ok=True)

print(f'start {exp_name}')
loss = []
for i in range(10):
    print(f'{i}-th iteration')
    gan = GAN(data_dim=data_dim, eps=eps)
    gan.dist_init(setting='mu', true_mean=0, out_mean=5)
    gan.data_init(data_size=data_size, mc_ratio=3)
    gan.model_init()
    gan.optimizer_init(lr_d=1, lr_g=0.02, decay_par=0.4,
                       reg_d=6e-5, reg_g=5e-5, update_D_iter=1)
    gan.fit(optim_iter=3000)
    loss.append(gan.score(100))
    # plt.subplots()
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(gan.l2_loss)
    plt.xlabel('optim step')
    plt.ylabel('l2 loss')
    plt.title(f'd={gan.data_dim}, n={data_size}, eps={eps}')
    plt.subplot(1, 2, 2)
    plt.plot(np.array(gan.G_record))
    plt.xlabel('optim step')
    plt.ylabel('estimated value of mean')
    plt.title(f'd={gan.data_dim}, n={data_size}, eps={eps}')
    plt.savefig(f'{dir}/{exp_name}/{i}.png')
loss = np.array(loss)

with open(f'{dir}/loss_mean.txt', 'w') as f:
    f.write(f'{np.mean(loss)},\n')
    
with open(f'{dir}/loss_std.txt', 'w') as f:   
    f.write(f'{np.std(loss)},\n')
    
print(f'end {exp_name}')
