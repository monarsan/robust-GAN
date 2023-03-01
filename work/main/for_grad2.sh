for dim in 10 25 50 75 100
do
    for lr_d in 0.1 0.25 0.5 0.75 1
    do
        python3 gd_script.py $dim 1000 0.1 dim100_find_hyper_gd 0.95 1 0.0001 1 $lr_d 3000
    done
done