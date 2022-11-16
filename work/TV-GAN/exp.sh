n="1000"
eps="0.1"
data_dim="2"
mu="0"
mu_out="0.5"
par_reg1="0.0005"
lr="0.0001"
decay_par="0.5"
exper_iter="10"
optim_iter="5000"
mm_iter="1"
optim_method="BFGS"  #Nelder-Mead, SLSQP BFGS
init_loc="0"
learn_par_u="0.1"

#Do python
for learn_par_u in 1 0.1 0.01 0.001 0.0001
do
    python3 ./multivar.py "$n" "$eps" "$data_dim" "$mu" "$mu_out" "$exper_iter" "$optim_iter" "$mm_iter" "$optim_method" "$decay_par" "$par_reg1" "$lr" "$init_loc" "$learn_par_u"
done