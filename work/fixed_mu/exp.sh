n="3000"
eps="0.1"
data_dim="10"
mu="0"
mu_out="6"
par_reg1="0.005"
lr="0.01"
decay_par="0.5"
exper_iter="2"
optim_iter="500"
mm_iter="1"
optim_method="BFGS"  #Nelder-Mead, SLSQP BFGS
init_loc="0"
exper_name=''
cov0='kendall'
u0='mle'


lr="0.08"
exper_name='lr=008 out6 decay05'
python3 ./mat_cholesy.py "$n" "$eps" "$data_dim" "$mu" "$mu_out" "$exper_iter" \
"$optim_iter" "$mm_iter" "$optim_method" "$decay_par" "$par_reg1" "$lr" "$init_loc" "$exper_name" \
"$cov0" "$u0"

#Do python

# python3 ./mat_cholesy.py "$n" "$eps" "$data_dim" "$mu" "$mu_out" "$exper_iter" "$optim_iter" "$mm_iter" "$optim_method" "$decay_par" "$par_reg1" "$lr" "$init_loc"
#     python3 ./mat_cholesy.py "$n" "$eps" "$data_dim" "$mu" "$mu_out" "$exper_iter" "$optim_iter" "$mm_iter" "$optim_method" "$decay_par" "$par_reg1" "$lr" "$init_loc"

#python3 ./fixed_mu.py "$n" "$eps" "$data_dim" "$mu" "$mu_out" "$exper_iter" "$optim_iter" "$mm_iter" "$optim_method" "$decay_par" "$par_reg1" "$lr" "$init_loc"
# exper_name='mu-out1, sigma0 MLE, u0 MLE'
# cov0='mle'
# u0='mle'
# python3 ./mat_cholesy.py "$n" "$eps" "$data_dim" "$mu" "$mu_out" "$exper_iter" \
# "$optim_iter" "$mm_iter" "$optim_method" "$decay_par" "$par_reg1" "$lr" "$init_loc" "$exper_name" \
# "$cov0" "$u0"

# mu_out="6"
# exper_name='mu-out6, sigma0 MLE, u0 MLE'
# python3 ./mat_cholesy.py "$n" "$eps" "$data_dim" "$mu" "$mu_out" "$exper_iter" \
# "$optim_iter" "$mm_iter" "$optim_method" "$decay_par" "$par_reg1" "$lr" "$init_loc" "$exper_name" \
# "$cov0" "$u0"


# mu_out="1"
# exper_name='mu-out1, sigma0 MLE, u0 kendall'
# cov0='mle'
# u0='kendall'
# python3 ./mat_cholesy.py "$n" "$eps" "$data_dim" "$mu" "$mu_out" "$exper_iter" \
# "$optim_iter" "$mm_iter" "$optim_method" "$decay_par" "$par_reg1" "$lr" "$init_loc" "$exper_name" \
# "$cov0" "$u0"

# mu_out="6"
# exper_name='mu-out6, sigma0 MLE, u0 kendall'
# python3 ./mat_cholesy.py "$n" "$eps" "$data_dim" "$mu" "$mu_out" "$exper_iter" \
# "$optim_iter" "$mm_iter" "$optim_method" "$decay_par" "$par_reg1" "$lr" "$init_loc" "$exper_name" \
# "$cov0" "$u0"


# mu_out="1"
# exper_name='mu-out1, sigma0 kendall, u0 MLE'
# cov0='kendall'
# u0='mle'
# python3 ./mat_cholesy.py "$n" "$eps" "$data_dim" "$mu" "$mu_out" "$exper_iter" \
# "$optim_iter" "$mm_iter" "$optim_method" "$decay_par" "$par_reg1" "$lr" "$init_loc" "$exper_name" \
# "$cov0" "$u0"

# mu_out="6"
# exper_name='mu-out6, sigma0 kendall, u0 MLE'
# python3 ./mat_cholesy.py "$n" "$eps" "$data_dim" "$mu" "$mu_out" "$exper_iter" \
# "$optim_iter" "$mm_iter" "$optim_method" "$decay_par" "$par_reg1" "$lr" "$init_loc" "$exper_name" \
# "$cov0" "$u0"

# n="500"
# mu_out="6"
# exper_name='mu-out1, sigma0 kendall, u0 kendall, add reg, n 500'
# python3 ./mat_cholesy.py "$n" "$eps" "$data_dim" "$mu" "$mu_out" "$exper_iter" \
# "$optim_iter" "$mm_iter" "$optim_method" "$decay_par" "$par_reg1" "$lr" "$init_loc" "$exper_name" \
# "$cov0" "$u0"

# n="1500"
# mu_out="6"
# exper_name='mu-out1, sigma0 kendall, u0 kendall, add reg, n 1500'
# python3 ./mat_cholesy.py "$n" "$eps" "$data_dim" "$mu" "$mu_out" "$exper_iter" \
# "$optim_iter" "$mm_iter" "$optim_method" "$decay_par" "$par_reg1" "$lr" "$init_loc" "$exper_name" \
# "$cov0" "$u0"

# n="2500"
# mu_out="6"
# exper_name='mu-out1, sigma0 kendall, u0 kendall, add reg, n 2500'
# python3 ./mat_cholesy.py "$n" "$eps" "$data_dim" "$mu" "$mu_out" "$exper_iter" \
# "$optim_iter" "$mm_iter" "$optim_method" "$decay_par" "$par_reg1" "$lr" "$init_loc" "$exper_name" \
# "$cov0" "$u0"

# n="3500"
# mu_out="6"
# exper_name='mu-out1, sigma0 kendall, u0 kendall, add reg, n 3500'
# python3 ./mat_cholesy.py "$n" "$eps" "$data_dim" "$mu" "$mu_out" "$exper_iter" \
# "$optim_iter" "$mm_iter" "$optim_method" "$decay_par" "$par_reg1" "$lr" "$init_loc" "$exper_name" \
# "$cov0" "$u0"


# mu_out="6"
# exper_name='mu-out6, sigma0 mle, u0 kendall, add reg, eps 0.05'
# python3 ./mat_cholesy.py "$n" "$eps" "$data_dim" "$mu" "$mu_out" "$exper_iter" \
# "$optim_iter" "$mm_iter" "$optim_method" "$decay_par" "$par_reg1" "$lr" "$init_loc" "$exper_name" \
# "$cov0" "$u0"


# mu_out="1"
# for n in  50 200 400 600 800 1000 
# do
#     python3 ./mat_cholesy.py "$n" "$eps" "$data_dim" "$mu" "$mu_out" "$exper_iter" "$optim_iter" "$mm_iter" "$optim_method" "$decay_par" "$par_reg1" "$lr" "$init_loc"
# done

# lr='0.1'
# mu_out="6"
# for n in  50 200 400 600 800 1000 
# do
#     python3 ./mat_cholesy.py "$n" "$eps" "$data_dim" "$mu" "$mu_out" "$exper_iter" "$optim_iter" "$mm_iter" "$optim_method" "$decay_par" "$par_reg1" "$lr" "$init_loc"
# done
# mu_out="1"
# for n in  50 200 400 600 800 1000 
# do
#     python3 ./mat_cholesy.py "$n" "$eps" "$data_dim" "$mu" "$mu_out" "$exper_iter" "$optim_iter" "$mm_iter" "$optim_method" "$decay_par" "$par_reg1" "$lr" "$init_loc"
# done






# for mu_out in  1 1 1 1 1 1
# do
#     python3 ./mat_cholesy.py "$n" "$eps" "$data_dim" "$mu" "$mu_out" "$exper_iter" "$optim_iter" "$mm_iter" "$optim_method" "$decay_par" "$par_reg1" "$lr" "$init_loc"
# done

# for lr in  0.1 0.01 0.001
# do
#     for mu_out in  6 6 6 6 6
#     do
#         python3 ./mat_cholesy.py "$n" "$eps" "$data_dim" "$mu" "$mu_out" "$exper_iter" "$optim_iter" "$mm_iter" "$optim_method" "$decay_par" "$par_reg1" "$lr" "$init_loc"
#     done
# done




# for lr in 0.01 0.001 0.0001
# do
#     for decay_par in 0.1 0.3 0.5
#     do
#         for mm_iter in 1 2 3
#         do
#             python3 ./mat_cholesy.py "$n" "$eps" "$data_dim" "$mu" "$mu_out" "$exper_iter" "$optim_iter" "$mm_iter" "$optim_method" "$decay_par" "$par_reg1" "$lr" "$init_loc"
#         done
#     done

# done

# for n in 1000 5000 10000
# do
#     python3 ./mat_cholesy.py "$n" "$eps" "$data_dim" "$mu" "$mu_out" "$exper_iter" "$optim_iter" "$mm_iter" "$optim_method" "$decay_par" "$par_reg1" "$lr" "$init_loc"
# done