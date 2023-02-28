

for lr_g in 0.1 0.25 0.5 0.75 1
do
    python3 grad_paper.py 10 1000 0.1 dim100_find_hyper 0.95 1 0.0001 $lr_g 1 3000
done

for lr_g in 0.1 0.25 0.5 0.75 1
do
    python3 grad_paper.py 25 1000 0.1 dim100_find_hyper 0.95 1 0.0001 $lr_g 1 3000
done

for lr_g in 0.1 0.25 0.5 0.75 1
do
    python3 grad_paper.py 50 1000 0.1 dim100_find_hyper 0.95 1 0.0001 $lr_g 1 1000
done

for lr_g in 0.1 0.25 0.5 0.75 1
do
    python3 grad_paper.py 75 1000 0.1 dim100_find_hyper 0.95 1 0.0001 $lr_g 1 1000
done

for lr_g in 0.1 0.25 0.5 0.75 1
do
    python3 grad_paper.py 100 1000 0.1 dim100_find_hyper 0.95 1 0.0001 $lr_g 1 1000
done