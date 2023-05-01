from gan_torch import Mu
import optuna
import numpy as np
from optuna.visualization import matplotlib as optuna_plot
import matplotlib.pyplot as plt
import os


tuning_name = 'naive-tuning'
data_dim = 5
rcd_dir = f'optuna/{tuning_name}/'
os.makedirs(rcd_dir, exist_ok=True)


def objective(trial):
    lr_d = trial.suggest_float('lr_d', 0.001, 1, log=True)
    lr_g = trial.suggest_float('lr_g', 0.001, 1, log=True)
    results = []
    for _ in range(5):
        gan = Mu(data_dim, 0.1, 'cpu')
        true_mean = np.zeros(data_dim)
        out_mean = np.ones(data_dim) * 5
        gan.dist_init(true_mean, out_mean)
        gan.data_init(1000, 25)
        gan.model_init()
        gan.optimizer_init(lr_d, lr_g, 5, 1)
        gan.fit(1500)
        results.append(gan.mean_err_record[-1])
    return np.mean(np.array(results))


if __name__ == "__main__":
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=400)
    
    print(study.best_params)
    with open(f'{rcd_dir}best_params.txt', mode='w') as f:
        f.write(str(study.best_params))
    
    # save fig
    fig = optuna_plot.plot_parallel_coordinate(study, params=["lr_d", "lr_g"])
    fig = fig.get_figure()
    fig.savefig(f'{rcd_dir}parallel_coordinate.png')
    fig2 = optuna_plot.plot_contour(study, params=["lr_d", "lr_g"])
    fig2 = fig2.get_figure()
    fig2.savefig(f'{rcd_dir}plot_contour.png')