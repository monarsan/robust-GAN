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
parser.add_argument('decay_par')
parser.add_argument('l_smooth')
parser.add_argument('grad_clip')

args = parser.parse_args()
data_dim = int(args.dim)
data_size = int(args.n)
eps = float(args.eps)
eps_int = int(100 * eps)
decay_par = float(args.decay_par)
l_smooth = float(args.l_smooth)
grad_clip = float(args.grad_clip)
dir = 'images_for_grad_dim100_change_decay_clip'

exp_name = f'{args.exp_name}/decay{decay_par * 10}clip{grad_clip * 1e4}'
os.makedirs(f'{dir}/{exp_name}', exist_ok=True)

print(f'start {exp_name}')
loss = []
for i in range(5):
    print(f'{i}-th iteration')
    gan = GAN(data_dim=data_dim, eps=eps)
    gan.dist_init(setting='mu', true_mean=5, out_mean=0)
    gan.data_init(data_size=data_size, mc_ratio=3)
    gan.model_init()
    gan.optimizer_init(lr_d=1, lr_g=1, decay_par=decay_par,
                       reg_d=6e-5, reg_g=5e-4, update_D_iter=1, 
                       l_smooth=l_smooth, grad_clip=grad_clip)
    gan.fit(optim_iter=1000, verbose=True)
    loss.append(gan.score(100))
    # plt.subplots()
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 3, 1)
    plt.plot(gan.l2_loss)
    plt.xlabel('optim step')
    plt.ylabel('l2 loss')
    plt.title(f'd={gan.data_dim}, n={data_size}, eps={eps}')
    plt.subplot(1, 3, 2)
    plt.plot(np.array(gan.G_record))
    plt.xlabel('optim step')
    plt.ylabel('estimated value of mean')
    plt.title(f'd={gan.data_dim}, n={data_size}, eps={eps}')
    plt.subplot(1, 3, 3)
    plt.plot(gan.D_data_record, label='target')
    plt.plot(gan.D_contami_record,label='contami')
    plt.plot(gan.D_z_record, label='z')
    plt.legend()
    plt.savefig(f'{dir}/{exp_name}/{i}.png')
loss = np.array(loss)

with open(f'{dir}/loss_mean.txt', 'a') as f:
    f.write(f'{np.mean(loss)},\n')
    
with open(f'{dir}/loss_std.txt', 'a') as f:   
    f.write(f'{np.std(loss)},\n')
    
print(f'end {exp_name}')
