from gan import gan as GAN
import optuna
import numpy as np
from optuna.visualization import matplotlib as optuna_plot
import matplotlib.pyplot as plt


data_dim = 2


def objective(trial):
    lr_d = trial.suggest_float('lr_d', 0.01, 1, log=True)
    lr_g = trial.suggest_float('lr_g', 0.1, 1, log=True)
    # reg_d = trial.suggest_float('reg_d', 1e-7, 1e-4, log=True)
    # reg_g = trial.suggest_float('reg_g', 1e-7, 1e-4, log=True)
    # decay_par_D = trial.suggest_float('decay_par_D', 0.1, 1, step=0.1)
    # decay_par_G = trial.suggest_float('decay_par_G', 0.1, 1, step=0.1)
    # update_D_iter = trial.suggest_int('update_D_iter', 1, 3, step=1)
    # l_smooth = trial.suggest_float('label_smooth', 0.8, 1.2, log=True)
    # decay_par = trial.suggest_float('decay_par', 0.4, 0.6, log=True)
    results = []
    for _ in range(20):
        gan = GAN(data_dim, 0.1)
        gan.dist_init('sigma', 0, 5, sigma_setting='ar')
        gan.data_init(1000, 3)
        gan.model_init(D_init_option='mle', G_init_option='kendall')
        gan.optimizer_init(lr_d, lr_g, 0.95, 1e-4, 1e-4,
                           update_D_iter=1, is_mm_alg=False)
        gan.fit(500)
        results.append(gan.l2_loss)
    results = np.array(results)
    results = results - results[:, 0, np.newaxis]
    results = np.mean(results, axis=1)
    print(results)
    for i in range(len(results)):
        trial.report(-results[i], step=i)
    return -np.array(results)[-1]


if __name__ == "__main__":
    study = optuna.create_study()
    study.optimize(objective, n_trials=100)
    print(study.best_params)
    fig = optuna_plot.plot_parallel_coordinate(study, params=["lr_d", "lr_g"])
    fig = fig.get_figure()
    fig.savefig(f'optuna_plot/sigme{data_dim}-1.png')
    fig2 = optuna_plot.plot_contour(study, params=["lr_d", "lr_g"])
    fig2 = fig2.get_figure()
    fig2.savefig(f'optuna_plot/sigma{data_dim}-2.png')
    fig3 = optuna_plot.plot_intermediate_values(study)
    fig3 = fig3.get_figure()
    fig3.savefig(f'optuna_plot/sigma{data_dim}-3.png')
