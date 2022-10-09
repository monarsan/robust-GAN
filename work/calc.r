
n <- 300
m <- 3*n   ## #samples by Metropolis-Hastings (for Gaussian distribution, )

eps <- 0.1                       ## epsilon
par_mu <- 0; par_sd <- sqrt(1)   ## ターゲット   N(par_mu, par_sd^2)
out_mu <- 5; out_sd <- sqrt(1)   ## 外れ値の分布 N(out_mu, out_sd^2)

true_alpha <- c(par_mu, par_sd^2)

exper_itr <- 10  ## iteration of experiments with different data (実験の繰り返し数)
iterate <- 500   ## iteration of outer loop (outer loopの回数)
L <- 100         ## iteration of inner loop (inner loop(MM-alg)の更新回数)
learn_par <- 1   ## learning parameter for alpha 
dicay_par <- 0.5 ## learning rate: order


sigmoid <- function(t){ ## sigmoid関数
  return(1/(1+exp(-t)))
}
deriv_sigmoid <- function(t){ ## sigmoidの微分
  return(sigmoid(t) * (1-sigmoid(t)))
}
g_up <- function(t,s){  ## MM-alg 上界
  sigmoid(s) + deriv_sigmoid(s) * (t-s) + (t-s)^2/20
}
g_lo <- function(t,s){  ## MM-alg 下界
  sigmoid(s) + deriv_sigmoid(s) * (t-s) - (t-s)^2/20
}



res <- list()
for (jj in 1:exper_itr){
  cat('== ',jj,'/',exper_itr,' ==\n')
  n_in <- sum(runif(n)<1-eps)
  n_out <- n - n_in
  x <- rnorm(n_in ,mean=par_mu, sd=par_sd)
  o <- rnorm(n_out,mean=out_mu, sd=out_sd)
  dat <- c(x,o); dat <- dat[sample(n)]      ## 観測データ
  ## initial alpha; alpha = (mu, v) of Gaussian
  alpha_hist <- c()
  alpha <- c(mean(dat),mad(dat)^2)         ## 期待値と分散の推定値の初期値
  for (ii in 1:iterate){
    ## if(ii %% 20==0){print(ii)}
    ## MMアルゴリズム
    z <- rnorm(m, mean=alpha[1], sd=sqrt(alpha[2]))
    major_func <- function(par, past_par){  ## beta: beta^T t(x), t(x)=(x,x^2). cgtidoc
      new_beta <- par[1:2]; new_b <- par[3]; beta <- past_par[1:2]; b <- past_par[3]
      A <- mean(g_lo(    cbind(z,z^2) %*% new_beta - new_b,     cbind(z,z^2) %*% beta - b))
      B <- mean(g_up(cbind(dat,dat^2) %*% new_beta - new_b, cbind(dat,dat^2) %*% beta - b))
      return(A-B)
    }
    l <- 0; par <- rnorm(3,sd=0.1) ##  par=(beta[1], beta[2], b)
    while(l<L){ ## MMアルゴリズム update of beta    
      op <- optim(c(0,0,0), major_func, past_par=par, control = list(fnscale = -1))
      par <- op$par
      ## print(op$value)
      l <- l+1
    }
    alpha_m <- alpha[1]; alpha_v <- alpha[2]
    mgrad <- t(rbind((z-alpha_m)/alpha_v, ((z-alpha_m)^2-alpha_v)/(2*alpha_v)))
    sig_ <- c(sigmoid(cbind(z,z^2) %*% par[1:2]-par[3]))

    ## 勾配法 for alpha
    tmp_alpha <- alpha - learn_par/ii^dicay_par * colMeans(mgrad * sig_)
    discount <- 0.5; cnt <- 1
    while(tmp_alpha[2]<0){ ## 分散パラメータが負にならないように line search を調整
      tmp_alpha <- alpha - learn_par/ii^dicay_par * discount^cnt * colMeans(mgrad * sig_)
      cnt <- cnt+1
    }
    alpha <- tmp_alpha
    alpha_hist <- rbind(alpha_hist,alpha)
    ##print(alpha)
  }
  res[[jj]] <- alpha_hist   ## 結果
}

## 保存
save(list=ls(), file='result_norm.rda')


