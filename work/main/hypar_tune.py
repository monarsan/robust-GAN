from gan import gan as GAN
import optuna
import joblib

def objective(trial):
    from gan import gan as GAN
    import numpy as np
    
    lr_d = trial.suggest_float('lr_d', 1e-4, 1, log=True)
    lr_g = trial.suggest_float('lr_g', 1e-4, 1e-1, log=True)
    decay_par = trial.suggest_float('decay_par', 0.1, 0.5, step = 0.1)
    reg_d = trial.suggest_float('reg_d', 1e-6, 1, log=True)
    results= []
    for _ in range(10):
        gan = GAN(10, 0.1)
        gan.dist_init('mu', 0, 5)
        gan.data_init(1000)
        gan.model_init()
        gan.optimizer_init(lr_d, lr_g, decay_par, reg_d, 0, 3000, 1)
        gan.fit(1000)
        results.append(gan.score(10))
    return np.mean(np.array(results))


if __name__ == "__main__":
    study = optuna.create_study()
    study.optimize(objective, n_trials=100)
    print(study.best_params)
