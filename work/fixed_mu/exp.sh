n="1000"
eps="0.1"
data_dim="15"
mu="0"
mu_out="6"
par_reg1="0.0005"
lr="0.005"
decay_par="0.3"
exper_iter="1"
optim_iter="100"
mm_iter="1"
optim_method="BFGS"  #Nelder-Mead, SLSQP BFGS
init_loc="0"


#Do python
for decay_par in 0.1
do
    python3 ./mat_cuda.py "$n" "$eps" "$data_dim" "$mu" "$mu_out" "$exper_iter" "$optim_iter" "$mm_iter" "$optim_method" "$decay_par" "$par_reg1" "$lr" "$init_loc"
done
