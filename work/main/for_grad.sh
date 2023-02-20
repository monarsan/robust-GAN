for decay in 0.3 0.4 0.5 0.6 0.7
do
    python3 grad_paper.py 100 1000 0.1 dim100_decay $decay 1 0.1
done

for clip in 0.1 0.01 0.001 0.0001
do
    python3 grad_paper.py 100 1000 0.1 dim100 0.4 1 $clip
done
