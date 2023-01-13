from gan import gan as GAN
import numpy as np

"""
# create instance
gan = gan()

gan.dist_init('mu', ture mean, out mean)

gan.data_init(data_size, mc_size)

gan.model_init()

gan.optimizer_init(lr_d, lr_g, decay_par, reg_d, reg_g, mm_iter, l_smooth)

gan.fit(optim_iter)

gan.score(average_iter)
"""

#mu
#value: 0.05404253926886985.{'lr_d': 0.13745373371541617, 'lr_g': 0.020714684193157522, 'decay_par': 0.4, 'mm_iter': 2
#sigma
#{'lr_d': 0.2924093291126133, 'lr_g': 0.057221485403883464, 'decay_par': 0.12865864552374684, 'l-smooth': 0.9993740060005327}.
results = []
loss = []
for i in range(10):
    gan = GAN(10, 0.1)
    gan.dist_init('sigma', 0, 5)
    gan.data_init(1000)
    gan.model_init()
    gan.optimizer_init(0.3, 0.057, 0.13, 0, 0, 1, 1)
    gan.fit(100)
    gan.record_wandb(title = str(i))
#     score = gan.score(10)
#     results.append(score)
#     print(score)
#     loss.append(gan.l2_loss)
# print('mean')
# print(np.mean(np.array(results)))
# np.save('l2-loss.npy',np.array(loss))