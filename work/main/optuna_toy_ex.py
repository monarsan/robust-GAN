from gan import gan as GAN
import optuna
import numpy as np


def objective(trial):
    lr_d = trial.suggest_float('lr_d', 0.00001, 1, log=True)
    lr_g = trial.suggest_float('lr_g', 0.00001, 1, log=True)
    reg_d = trial.suggest_float('reg_d', 1e-7, 1e-4, log=True)
    reg_g = trial.suggest_float('reg_g', 1e-7, 1e-4, log=True)
    decay_par_D = trial.suggest_float('decay_par_D', 0.1, 1, step=0.1)
    decay_par_G = trial.suggest_float('decay_par_G', 0.1, 1, step=0.1)
    update_D_iter = trial.suggest_int('update_D_iter', 1, 3, step=1)
    # l_smooth = trial.suggest_float('label_smooth', 0.8, 1.2, log=True)
    # decay_par = trial.suggest_float('decay_par', 0.4, 0.6, log=True)
    results = []
    for _ in range(10):
        gan = GAN(2, 0.1)
        gan.dist_init('sigma', 0, 5, sigma_setting='ar')
        gan.data_init(1000, 3, is_scaling=True)
        gan.model_init(D_init_option='mle', G_init_option='kendall')
        gan.optimizer_init(lr_d, lr_g, decay_par_D, decay_par_G, reg_d, reg_g,
                           update_D_iter=update_D_iter, is_mm_alg=False)
        gan.fit(5)
        score = gan.score_from_init('l2') + gan.score_from_init('fro')
        results.append(0.5 * score)
    return -np.mean(np.array(results))


if __name__ == "__main__":
    study = optuna.create_study()
    study.optimize(objective, n_trials=5)
    print('==== best params ====')
    print(study.best_params)
    print('==== Importance ====')
    print(optuna.importance.get_param_importances(study))
