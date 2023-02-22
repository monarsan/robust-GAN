

for clip in 0.1 0.01 0.001 0.0001
do
    python3 grad_paper.py 100 1000 0.1 dim100 0.7 1 $clip
done
