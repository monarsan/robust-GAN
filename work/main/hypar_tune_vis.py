from gan import gan as GAN
import optuna
import numpy as np
from optuna.visualization import matplotlib as optuna_plot
import matplotlib.pyplot as plt

data_dim = 50
def objective(trial):
    lr_d = trial.suggest_float('lr_d', 0.1, 1, log=True)
    lr_g = trial.suggest_float('lr_g', 0.5, 2, log=True)
    # reg_d = trial.suggest_float('reg_d', 0.000001, 0.0001, log=True)
    # reg_g = trial.suggest_float('reg_g', 0.000001, 0.0001, log=True)
    # optim_step = trial.suggest_int('optim_step', 200, 2000, step=200)
    # decay_par_D = trial.suggest_float('decay_par_D', 0.01, 0.15, log=True)
    # lr_d = trial.suggest_float('lr_d', 0.1, 1, log=True)
    # lr_g = trial.suggest_float('lr_g', 0.001, 0.5, log=True)
    # reg_d = trial.suggest_float('reg_d', 0.000001, 0.001, log=True)
    # reg_g = trial.suggest_float('reg_d', 0.000001, 0.01, log=True)
    # optim_step = trial.suggest_int('optim_step', 200, 2000, step=200)
    # decay_par = trial.suggest_float('decay_par', 0.1, 1, log=True)    
    results = []
    for _ in range(5):
        gan = GAN(data_dim, 0.1)
        gan.dist_init('mu', 0, 5)
        gan.data_init(1000, 3)
        gan.model_init()
        gan.optimizer_init(lr_d, lr_g, 0.95, 1e-5, 1e-5, 5, 1, False, 0.1)
        gan.fit(3000, 1e-7)
        results.append(gan.score_from_init())
    return -np.mean(np.array(results))


if __name__ == "__main__":
    study = optuna.create_study()
    study.optimize(objective, n_trials=25)
    print(study.best_params)
    fig = optuna_plot.plot_parallel_coordinate(study, params=["lr_d", "lr_g"])
    fig = fig.get_figure()
    fig.savefig(f'optuna_plot/p{data_dim}-1.png')
    fig2 = optuna_plot.plot_contour(study, params=["lr_d", "lr_g"])
    fig2 = fig2.get_figure()
    fig2.savefig(f'optuna_plot/p{data_dim}-2.png')