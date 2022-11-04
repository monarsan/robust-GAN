n="1000"
eps="0.1"
data_dim="10"
mu="5"
mu_out="0"
par_reg1="0.0005"
lr="0.05"
decay_par="0.5"
exper_iter="10"
optim_iter="500"
mm_iter="1"
optim_method="BFGS"  #Nelder-Mead, SLSQP BFGS
init_loc="0"


#Do python
for n in 1000 2000 3000 5000
do
    python3 ./multivar.py "$n" "$eps" "$data_dim" "$mu" "$mu_out" "$exper_iter" "$optim_iter" "$mm_iter" "$optim_method" "$decay_par" "$par_reg1" "$lr" "$init_loc"
done

