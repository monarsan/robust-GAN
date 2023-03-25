import numpy as np
import matplotlib.pyplot as plt
from gan import gan as GAN


optim_iter = 950
for i in range(10):
    gan = GAN(5, 0.2)
    gan.dist_init('sigma', 0, 6, sigma_setting='ar')
    gan.data_init(50000, 1)
    gan.model_init(D_init_option='random', G_init_option='kendall')
    gan.optimizer_init(lr_d=1, lr_g=0.1, decay_g=0.2, reg_g=1e-4, reg_d=1e-4,
                       update_D_iter=5, is_mm_alg=False, decay_d=1, grad_clip=1e-3, lr_schedule='step', step=200)
    gan.fit(optim_iter, verbose=True)
    gan.record_npy(rcd_dir='record', rcd_name=f'est_sig_dim{gan.data_dim}_{i}')
    gan.plot(fig_scale=1)
    plt.show()
    
    gan = GAN(10, 0.2)
    gan.dist_init('sigma', 0, 6, sigma_setting='ar')
    gan.data_init(50000, 1)
    gan.model_init(D_init_option='random', G_init_option='kendall')
    gan.optimizer_init(lr_d=1, lr_g=0.1, decay_g=0.2, reg_g=1e-4, reg_d=1e-4,
                       update_D_iter=5, is_mm_alg=False, decay_d=1, grad_clip=1e-3, lr_schedule='step', step=200)
    gan.fit(optim_iter, verbose=True)
    gan.record_npy(rcd_dir='record', rcd_name=f'est_sig_dim{gan.data_dim}_{i}')
    gan.plot(fig_scale=1)
    plt.show()
    
    gan = GAN(25, 0.2)
    gan.dist_init('sigma', 0, 6, sigma_setting='ar')
    gan.data_init(50000, 1)
    gan.model_init(D_init_option='random', G_init_option='kendall')
    gan.optimizer_init(lr_d=1, lr_g=0.1, decay_g=0.2, reg_g=1e-4, reg_d=1e-4,
                       update_D_iter=5, is_mm_alg=False, decay_d=1, grad_clip=1e-3, lr_schedule='step', step=200)
    gan.fit(optim_iter, verbose=True)
    gan.record_npy(rcd_dir='record', rcd_name=f'est_sig_dim{gan.data_dim}_{i}')
    gan.plot(fig_scale=1)
    plt.show()
    