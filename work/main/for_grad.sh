for eps in 0.02 0.05 0.1 0.15 0.2
do
    python3 grad_paper.py 2 4000 $eps change_eps_dim2
done

for dim in 2 4 6 8 10 15 20 25
do
    python3 grad_paper.py $dim 1000 0.1 change_dim
done


for n in 50 100 200 400 700 1000
do
    python3 grad_paper.py 10 $n 0.1 change_n
done
