import numpy as np
import matplotlib.pyplot as plt
from gan import gan as GAN
import seaborn as sns
import optuna


def objective(trial):
    lr_d = trial.suggest_float('lr_d', 0.001, 10, log=True)
    lr_g = trial.suggest_float('lr_g', 0.0001, 1, log=True)
    results = []
    for i in range(10):
        gan = GAN(data_dim=2, eps=0.1)
        gan.dist_init(setting='mu', true_mean=5, out_mean=0)
        gan.data_init(data_size=1000, mc_ratio=3)
        gan.model_init()
        gan.optimizer_init(lr_d=lr_d, lr_g=lr_g, decay_par=0.4,
                           reg_d=6e-5, reg_g=5e-5, update_D_iter=1, is_mm_alg=False)
        gan.fit(optim_iter=5000, verbose=False)
        results.append(gan.score_from_init())
    return -np.mean(np.array(results))


if __name__ == "__main__":
    study = optuna.create_study()
    study.optimize(objective, n_trials=500)
    print(study.best_params)
