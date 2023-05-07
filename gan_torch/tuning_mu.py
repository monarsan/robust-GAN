from mu import Mu
import optuna
import numpy as np
from optuna.visualization import matplotlib as optuna_plot
import matplotlib.pyplot as plt
import os


tuning_name = 'normed-lr-reg'
data_dim = 50
rcd_dir = f'optuna/mu/{tuning_name}-{data_dim}/'
os.makedirs(rcd_dir, exist_ok=True)


def objective(trial):
    lr_d = trial.suggest_float('lr_d', 1e-2, 1, log=True)
    lr_g = trial.suggest_float('lr_g', 1e-2, 1, log=True)
    decay_d = trial.suggest_float('reg_d', 1e-5, 1e-1, log=True)
    decay_g = trial.suggest_float('reg_g', 1e-5, 1e-1, log=True)
    results = []
    for i in range(3):
        gan = Mu(data_dim, 0.1, 'cpu')
        true_mean = np.zeros(data_dim)
        out_mean = np.ones(data_dim) * 5
        gan.dist_init(true_mean, out_mean)
        gan.data_init(1000, 25)
        gan.model_init(D_model='quadratic')
        gan.optimizer_init(lr_d=lr_d, lr_g=lr_g, d_steps=5, g_steps=1, 
                           weight_decay_d=decay_d, weight_decay_g=decay_g,
                           scheduler='exp', gamma=0.995)
        gan.fit(500)
        intermediate_value = gan.mean_err_record[-10:].mean()
        trial.report(intermediate_value, i)
        if trial.should_prune():
            raise optuna.TrialPruned()
        results.append(intermediate_value)
    return np.mean(np.array(results))


if __name__ == "__main__":
    pruner = optuna.pruners.ThresholdPruner(upper=0.6)
    study = optuna.create_study(direction='minimize', pruner=pruner)
    study.optimize(objective, n_trials=300)
    
    print(study.best_params)
    with open(f'{rcd_dir}best_params.txt', mode='w') as f:
        f.write(str(study.best_params))
    
    # save fig
    fig = optuna_plot.plot_parallel_coordinate(study, params=["lr_d", "lr_g"])
    fig = fig.get_figure()
    fig.savefig(f'{rcd_dir}parallel_lr.png')
    fig2 = optuna_plot.plot_contour(study, params=["lr_d", "lr_g"])
    fig2 = fig2.get_figure()
    fig2.savefig(f'{rcd_dir}plot_lr.png')
    fig3 = optuna_plot.plot_parallel_coordinate(study, params=["reg_d", "reg_g"])
    fig3 = fig3.get_figure()
    fig3.savefig(f'{rcd_dir}parallel_reg.png')
    fig4 = optuna_plot.plot_contour(study, params=["reg_d", "reg_g"])
    fig4 = fig4.get_figure()
    fig4.savefig(f'{rcd_dir}plot_reg.png')