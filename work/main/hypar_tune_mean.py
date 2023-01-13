from gan import gan as GAN
import optuna


def objective(trial):
    import numpy as np
    lr_d = trial.suggest_float('lr_d', 0.1, 1, log=True)
    lr_g = trial.suggest_float('lr_g', 0.001, 0.5, log=True)
    reg_d = trial.suggest_float('reg_d', 0.000001, 0.001, log=True)
    reg_g = trial.suggest_float('reg_d', 0.000001, 0.01, log=True)
    optim_step = trial.suggest_int('optim_step', 200, 2000, step=200)
    decay_par = trial.suggest_float('decay_par', 0.1, 1, log=True)
    
    results = []
    for _ in range(10):
        gan = GAN(10, 0.1)
        gan.dist_init('mu', 0, 5)
        gan.data_init(1000, 3)
        gan.model_init()
        gan.optimizer_init(lr_d, lr_g, decay_par, reg_d, reg_g, 1, 1)
        gan.fit(optim_step)
        results.append(gan.score_from_init())
    for _ in range(10):
        gan = GAN(10, 0.1)
        gan.dist_init('mu', 5, 0)
        gan.data_init(1000, 3)
        gan.model_init()
        gan.optimizer_init(lr_d, lr_g, decay_par, reg_d, reg_g, 1, 1)
        gan.fit(optim_step)
        results.append(gan.score_from_init())
    return -np.mean(np.array(results))
#  {'lr_d': 0.8539778326380212, 'lr_g': 0.013628436913602222, 'reg_d': 9.493624361550345e-05, 'optim_step': 1200, 'decay_par': 0.3795451920285185}


if __name__ == "__main__":
    study = optuna.create_study()
    study.optimize(objective, n_trials=500)
    print(study.best_params)
