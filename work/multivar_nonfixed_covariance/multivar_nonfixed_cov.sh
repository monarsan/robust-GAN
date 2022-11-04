#chmod +x multivar_nonfixed_cov.sh
#hyper par
n="10"
eps="0.1"
data_dim="5"
mu="0"
mu_out="5"
exper_iter="10"
optim_iter="1"
mm_iter="2"
#Nelder-Mead, SLSQP
optim_method="SLSQP"
save_dir="tmp"
#Do python
mkdir -p "exp_multivar_nonfixed_cov/$save_dir"
nohup /home/yokoyama/robust-gan/bin/python3 ./multivar_nonfixed_cov.py "$n" "$eps" "$data_dim" "$mu" "$mu_out" "$exper_iter" "$optim_iter" "$mm_iter" "$optim_method" "$save_dir" &
