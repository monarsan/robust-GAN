import numpy as np
import numpy.linalg as LA
from libs.functions import *
from libs.create import *
from typing import List
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


class gan(object):
    def __init__(self, data_dim: int, eps: float) -> None:
        super().__init__()
        self.data_dim = data_dim
        self.eps = eps

    def dist_init(self, setting: str, true_mean: float, out_mean: float, sigma_setting='sparse') -> None:
        """This function define distribution
        Args:
            setting (str): define problem setting, estimate mu or estimate sigma
            true_mean (float): mean of target distribution
            out_mean (float): mean of outlier distribution
        """
        # todo if message of error of this type increase, refactor
        if setting not in ['mu', 'sigma']:
            raise NameError("setting must to be mu or sigma")
        if sigma_setting not in ['sparse', 'concentrate', 'ar']:
            raise NameError("check setting plz")
        self.setting = setting
        self.sigma_setting = sigma_setting
        self.true_mean = np.full(self.data_dim, true_mean)
        self.out_mean = np.full(self.data_dim, out_mean)
        if self.is_sigma_setting():
            if self.is_sparse_sigma():
                self.true_cov = create_sparse_cov(self.data_dim)
                self.out_cov = create_sparse_cov(self.data_dim)
            elif self.is_concentrate_sigma():
                self.true_cov = np.identity(self.data_dim)
                self.out_cov = np.identity(self.data_dim) * 0.1
            else:
                self.true_cov = np.identity(self.data_dim)
                self.out_cov = np.identity(self.data_dim)
        else:
            self.true_cov = np.identity(self.data_dim)
            self.out_cov = np.identity(self.data_dim)

    def data_init(self, data_size: int, mc_ratio=3) -> None:
        self.data_size = data_size
        self.mc_size = mc_ratio * self.data_size
        self.data, self.target_data, self.contami_data = create_norm_data(self.data_size, self.eps, self.true_mean,
                                                                          self.true_cov, self.out_mean, self.out_cov)
        self.median = np.median(self.data, axis=0)
        self.emperical_true_mean = np.mean(self.target_data, axis=0)

    def model_init(self, D_init_option='mle', G_init_option='kendall', D_init_scale=0.1) -> None:
        if self.is_mu_setting():
            self.D = np.random.normal(0, D_init_scale, 2 * self.data_dim)
            # tmp_b = np.random.normal(0, 0.1, self.data_dim)
            # tmp_a = -0.5 * tmp_b / self.median
            # self.D = np.concatenate([tmp_a, tmp_b], axis=0)
            self.G = np.median(self.data, axis=0)
        else:
            self.D = init_discriminator(self.data, D_init_option)
            if G_init_option == 'true':
                self.G = LA.cholesky(self.true_cov)
            else:
                self.G = init_covariance(self.data, G_init_option)
        self.bias = self._u(self._z()).mean(axis=0)
        self.D_record = [self.D]
        if self.is_sigma_setting():
            self.G_record = [self.G @ self.G.T]
        if self.is_mu_setting():
            self.G_record = [self.G]
        self.bias_record = [self.bias]
        self.D_data_record = [self._D(self.data).mean()]
        self.D_target_record = [self._D(self.target_data).mean()]
        self.D_contami_record = [self._D(self.contami_data).mean()]
        self.D_z_record = [self._D(self._z()).mean()]

    def _u(self, x: List[float]) -> List[float]:
        d = self.data_dim
        if self.is_mu_setting():
            return np.sum((x ** 2) * self.D[:d], axis=1)\
                + np.sum(x * self.D[d: 2 * d], axis=1)
        else:
            return sample_wise_vec_mat_vec(self.D.reshape(d, d), x)

    def _D(self, x: List[float]) -> List[float]:
        return sigmoid(self._u(x) - self.bias)

    def _z(self):
        if self.is_mu_setting():
            z = np.random.multivariate_normal(mean=self.G,
                                              cov=np.identity(self.data_dim),
                                              size=self.mc_size)
        else:
            z = np.random.multivariate_normal(mean=self.true_mean,
                                              cov=self.G @ self.G.T,
                                              size=self.mc_size)
        return z

    # todo: add default par after do optuna
    def optimizer_init(self, lr_d, lr_g, decay_par, reg_d=0, reg_g=0, update_D_iter=1, l_smooth=1, is_mm_alg=True, grad_clip=0.1):
        self.lr_d = lr_d
        self.lr_g = lr_g
        self.decay_par = decay_par
        self.reg_d = reg_d
        self.reg_g = reg_g
        self.update_D_iter = update_D_iter
        self.l_smooth = l_smooth
        self.is_mm_alg = is_mm_alg
        self.threshold = grad_clip

    def _est_cov(self):
        return self.G @ self.G.T

    # todo: add average epochs
    def fit(self, optim_iter, tol=1e-6, verbose=False):
        self.tol = tol * (self.data_dim ** 0.5)
        self.objective = [self._D(self._z()).mean() - self._D(self.data).mean()]
        self.optim_iter = optim_iter
        if self.is_sigma_setting():
            self.fro_loss = [
                LA.norm(self._est_cov() - self.true_cov, ord='fro')]
            self.l2_loss = [LA.norm(self._est_cov() - self.true_cov, ord=2)]
        else:
            self.l2_loss = [LA.norm(self.G - self.true_mean, ord=2)]
        for i in tqdm(range(optim_iter), disable=(not verbose)):
            self.iter = i
            self.z = self._z()
            self.D_data_record.append(self._D(self.data).mean())
            self.D_target_record.append(self._D(self.target_data).mean())
            self.D_contami_record.append(self._D(self.contami_data).mean())
            self.D_z_record.append(self._D(self.z).mean())
            # todo normalize input
            # Update D
            if self.is_mm_alg:
                if self.is_mu_setting():
                    self._mm_alg_mu()
                else:
                    self._mm_alg_sigma()
            else:
                self._update_u_via_GD()
            self.D_record.append(self.D)
            self.bias_record.append(self.bias)
            # Update G
            self.prev_G = self.G
            if self.is_mu_setting():
                self._GD_mu()
                self.l2_loss.append(LA.norm(self.G - self.true_mean, ord=2))
                self.G_record.append(self.G)
            else:
                self._GD_sigma()
                self.l2_loss.append(
                    LA.norm(self._est_cov() - self.true_cov, ord=2))
                self.fro_loss.append(
                    LA.norm(self._est_cov() - self.true_cov, ord='fro'))
                self.G_record.append(self.G @ self.G.T)
            converged = (self.iter >= self.optim_iter) or (LA.norm(self.prev_G - self.G, axis=0) < self.tol)
            if converged:
                print(f'converged at {self.iter} step')
                break

    # functions for optimization
    def _mm_alg_mu(self):
        data_dim = self.data_dim
        data = self.data
        z = self.z
        # todo normalize
        counter_mm = 0
        z_sq = z**2
        data_sq = data**2
        while counter_mm < self.update_D_iter:
            t0_z = self._u(z) - self.bias  # shape (m,)
            t0_data = self._u(data) - self.bias  # (n,)

            # 連立方程式の行列を求める Ax = b
            # ここからがMMアルゴリズムの計算
            A1 = -0.1 * (self.l_smooth * mean_outer_product(z_sq, z_sq) + mean_outer_product(data_sq, data_sq))\
                - self.reg_d * data_dim
            A2 = -0.1 * (self.l_smooth * mean_outer_product(z_sq, z) + mean_outer_product(data_sq, data))
            A3 = 0.1 * (self.l_smooth * z_sq.mean(axis=0) + data_sq.mean(axis=0))
            b1 = - (self.l_smooth * (deriv_sigmoid(t0_z) + t0_z / 10)[:, np.newaxis] * z_sq).mean(axis=0)\
                + ((deriv_sigmoid(t0_data) - t0_data / 10)[:, np.newaxis] * data_sq).mean(axis=0)

            A4 = -0.1 * (self.l_smooth * mean_outer_product(z, z_sq) + mean_outer_product(data, data_sq))
            A5 = -0.1 * (self.l_smooth * mean_outer_product(z, z) + mean_outer_product(data, data)) - self.reg_d * data_dim
            A6 = 0.1 * self.l_smooth * z.mean(axis=0) + 0.1 * data.mean(axis=0)
            b2 = -(self.l_smooth * (deriv_sigmoid(t0_z) + t0_z / 10)[:, np.newaxis] * z).mean(
                axis=0) + ((deriv_sigmoid(t0_data) - t0_data / 10)[:, np.newaxis] * data).mean(axis=0)

            A7 = -0.1 * self.l_smooth * z_sq.mean(axis=0) + 0.1 * data_sq.mean(axis=0)
            A8 = -0.1 * self.l_smooth * z.mean(axis=0) + 0.1 * data.mean(axis=0)
            A9 = np.full(1, - self.l_smooth * (0.1) - (0.1))
            b3 = np.array([self.l_smooth * np.mean(deriv_sigmoid(t0_z) - t0_z / 10, axis=0)
                           - np.mean(deriv_sigmoid(t0_data) - t0_data / 10, axis=0)])

            A_a = np.concatenate([A1, A2, A3[:, np.newaxis]], axis=1)
            A_b = np.concatenate([A4, A5, A6[:, np.newaxis]], axis=1)
            A_bias = np.concatenate([A7, A8, A9], axis=0)
            A = np.concatenate([A_a, A_b, A_bias[np.newaxis, :]], axis=0)
            b = np.concatenate([b1, b2, b3], axis=0)
            decayed_lr_d = self.lr_d / (self.iter + 1) ** self.decay_par
            new_par = LA.solve(A, b)
            self.D = self.D * (1 - decayed_lr_d) + decayed_lr_d * new_par[:-1]
            self.bias = self.bias * (1 - decayed_lr_d) + decayed_lr_d * new_par[-1]
            counter_mm += 1

    def _GD_mu(self):
        z = self.z
        mgrad = (z - self.G)
        sig_ = self._D(z)[:, np.newaxis]
        lr_tmp = self.lr_g / (self.iter + 1) ** self.decay_par
        # print(np.mean(mgrad * sig_, axis=0))
        grad = np.mean(mgrad * sig_, axis=0)  # (d,)
        grad = self.clip(grad, self.threshold)
        self.G = self.G - lr_tmp * np.mean(mgrad * sig_, axis=0) - self.reg_g / (self.iter + 1) ** self.decay_par * (self.G - self.median)
        self.objective.append(self._D(self.z).mean() - self._D(self.data).mean())

    # todo 変数名がめちゃくちゃ
    def _mm_alg_sigma(self):
        data_dim = self.data_dim
        data = self.data
        z = self.z
        zzT = sample_wise_outer_product(z, z)
        xxT = sample_wise_outer_product(data, data)
        A = np.zeros([data_dim, data_dim, data_dim, data_dim])
        A_bias_col = np.zeros([data_dim, data_dim])
        A_b = np.zeros([data_dim, data_dim])
        A_bias_row = np.zeros([data_dim, data_dim])
        counter_mm = 0
        while (counter_mm < self.update_D_iter):
            t0_z = self._u(z) - self.bias  # shape (m,)
            t0_data = self._u(data) - self.bias  # (n,)
            # todo この2重ループの計算量がおおい
            for i in range(data_dim):
                for k in range(data_dim):
                    A[i][k] = -0.1 * (self.l_smooth * np.mean(z[:, i][:, np.newaxis, np.newaxis] * z[:, k][:, np.newaxis, np.newaxis] * zzT, axis=0)
                                      + np.mean(data[:, i][:, np.newaxis, np.newaxis] * data[:, k][:, np.newaxis, np.newaxis] * xxT, axis=0))
                # reguralization for A
                A[i][i] -= np.identity(data_dim) * self.lr_d
                A_bias_col[i] = 0.1 * (self.l_smooth * np.mean(z[:, i][:, np.newaxis] * z, axis=0)
                                       + np.mean(data[:, i][:, np.newaxis] * data, axis=0))
                A_b[i] = -(self.l_smooth * np.mean(((deriv_sigmoid(t0_z) + t0_z / 10) * z[:, i])[:, np.newaxis] * z, axis=0)
                           - np.mean(((deriv_sigmoid(t0_data) - t0_data / 10) * data[:, i])[:, np.newaxis] * data, axis=0))
                A_bias_row[i] = 0.1 * (self.l_smooth * np.mean(z[:, i, np.newaxis]
                                       * z, axis=0) + np.mean(data[:, i, np.newaxis] * data, axis=0))
            bias_bias = -0.1 * (self.l_smooth + 1)
            b_bias = (self.l_smooth * np.mean(deriv_sigmoid(t0_z) + t0_z / 10,
                      axis=0) - np.mean(deriv_sigmoid(t0_data - t0_data / 10), axis=0))
            A_bias_col_reshaped = A_bias_col.reshape(data_dim**2, 1)
            A_bias_row_reshaped = A_bias_row.reshape(1, data_dim**2)
            A_b_reshaped = A_b.reshape(data_dim**2)
            A_reshaped = A.T.reshape(
                data_dim, data_dim**2, data_dim).T.reshape(data_dim**2, data_dim**2)
            A_concated = np.concatenate(
                [A_reshaped, A_bias_col_reshaped], axis=1)
            A_bias_row_concated = np.concatenate(
                [A_bias_row_reshaped, np.array(bias_bias)[np.newaxis, np.newaxis]], axis=1)
            A_ = np.concatenate([A_concated, A_bias_row_concated], axis=0)
            b = np.concatenate([A_b_reshaped, b_bias[np.newaxis]], axis=0)
            decayed_lr_d = self.lr_d / (self.iter + 1) ** self.decay_par
            new_par = LA.solve(A_, b)
            self.D = self.D * (1 - decayed_lr_d) + decayed_lr_d * new_par[:-1]
            self.bias = self.bias * (1 - decayed_lr_d) + decayed_lr_d * new_par[-1]
            counter_mm += 1

    # todo Use Adam

    def _GD_sigma(self):
        data_dim = self.data_dim
        z = self.z
        ABZ = (z @ self.D.reshape(data_dim, data_dim)) @ (self.G + self.G.T)
        sigma_grad = sample_wise_outer_product(ABZ, z)  # (m, d, d)
        sig_ = deriv_sigmoid(self._u(z @ self.G) - self.bias)[:, np.newaxis]  # (m,)
        tmp_alpha_v = self.G - self.lr_g / (self.iter + 1) ** self.decay_par * np.mean(sigma_grad * sig_[:, :, np.newaxis], axis=0)
        self.G = tmp_alpha_v
        self.objective.append(self._D(self.z).mean() - self._D(self.data).mean())

    def _update_u_via_GD(self):
        z = self.z
        deriv_sig_z = deriv_sigmoid(self._u(z) - self.bias)[:, np.newaxis]  # shape : (m,)
        deriv_sig_data = deriv_sigmoid(self._u(self.data) - self.bias)[:, np.newaxis]  # shape : (m,)
        if self.is_mu_setting():
            z_sq = z ** 2
            data_sq = self.data ** 2
            grad_z_a = np.mean(deriv_sig_z * z_sq, axis=0)  # shape : (m, d)
            grad_z_b = np.mean(deriv_sig_z * z, axis=0)
            grad_data_a = np.mean(deriv_sig_data * data_sq, axis=0)
            grad_data_b = np.mean(deriv_sig_data * self.data, axis=0)
            grad_a = grad_z_a - grad_data_a
            grad_b = grad_z_b - grad_data_b
            grad = np.concatenate([grad_a, grad_b], axis=0)
            
        elif self.is_sigma_setting():
            zzT = sample_wise_outer_product(z, z)
            xxT = sample_wise_outer_product(self.data, self.data)
            grad_z = np.mean(deriv_sig_z * zzT, axis=0)  # shape : (m, d, d)
            grad_data = np.mean(deriv_sig_data * xxT, axis=0)
            grad = grad_z - grad_data
        grad_bias = np.mean(deriv_sig_data, axis=0) - np.mean(deriv_sig_z, axis=0)
        decayed_lr_d = self.lr_d / (self.iter + 1) ** self.decay_par
        decayed_lr_d = 1
        self.D = self.D * (1 - self.reg_d) + decayed_lr_d * grad
        self.bias = self.bias + decayed_lr_d * grad_bias

    def clip(self, x, threshold):
        for i in range(len(x)):
            if x[i] > threshold:
                x[i] = threshold * x[i] / np.abs(x[i])
        return x
        
    # funcitons for desplaying score
    def score(self, average: int) -> float:
        return np.mean(np.array(self.l2_loss[-average:]), axis=0)

    def score_from_init(self) -> float:
        return self.l2_loss[0] - self.l2_loss[-1]

    def record_wandb(self, title=None):
        import wandb
        import pandas as pd
        config = dict(outlier_mu=self.out_mean,
                      covariance_setting="sparse")
        exper_name = title
        wandb.init(
            project="robust scatter estimation",
            entity="robust-gan",
            config=config
        )
        if not exper_name == '':
            wandb.run.name = exper_name
            wandb.run.save()
        dict_plot = dict()
        df = pd.DataFrame()
        df['step'] = np.arange(self.optim_iter + 1)
        df['l2-loss'] = self.l2_loss
        df['bias'] = self.bias_record
        df['objective'] = np.array(self.objective)
        if self.is_sigma_setting():
            df['D norm'] = LA.norm(np.array(self.D_record).reshape(
                self.optim_iter + 1, self.data_dim, self.data_dim), axis=(1, 2))
            df['fro-loss'] = self.fro_loss
            df['A11'] = np.array(self.D_record)[:, 0]
            df['A12'] = np.array(self.D_record)[:, 1]
        table = wandb.Table(data=df)
        dict_plot['l2 loss'] = wandb.plot.line(
            table, 'step', 'l2-loss', title='l2 loss')
        dict_plot['bias'] = wandb.plot.line(
            table, 'step', 'bias', title='bias')
        dict_plot['objective'] = wandb.plot.line(
            table, 'step', 'objective', title='objective')
        if self.is_sigma_setting():
            dict_plot['D norm'] = wandb.plot.line(
                table, 'step', 'D norm', title='D norm')
            dict_plot['fro loss'] = wandb.plot.line(
                table, 'step', 'fro-loss', title='fro loss')
            dict_plot['A11'] = wandb.plot.line_series(xs=df['step'],
                                                      ys=[df['A11'], df['A12']],
                                                      keys=['A11', 'A12'],
                                                      xname='steps')
        wandb.log(dict_plot)
        wandb.finish()

# setting
    def is_mu_setting(self):
        return self.setting == 'mu'

    def is_sigma_setting(self):
        return self.setting == 'sigma'

    def is_sparse_sigma(self):
        return self.sigma_setting == 'sparse'

    def is_concentrate_sigma(self):
        return self.sigma_setting == 'concentrate'
