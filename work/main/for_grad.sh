for eps in 0.05 0.1 0.15 0.2
do
    python3 grad_paper.py 5 10000 $eps change_eps
done


for n in 100 200 400 700 1000
do
    python3 grad_paper.py 10 $n 0.1 change_n
done


for dim in 25 50
do
    python3 grad_paper.py $dim 1000 0.1 change_dim    
done 

for eps in 0.05 0.1 0.15 0.2
do
    python3 grad_paper.py 10 10000 $eps change_eps_dim10
done