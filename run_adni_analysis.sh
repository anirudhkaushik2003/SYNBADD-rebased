#!/bin/bash

# Create a new tmux session named "adni_analysis"
tmux new-session -d -s adni_analysis

# Send commands to the tmux session
tmux send-keys -t adni_analysis "cd /home2/anirudhkaushik/SYNBADD" C-m
tmux send-keys -t adni_analysis "conda activate pytorch" C-m

# Run the first fold
tmux send-keys -t adni_analysis "python adni_baseline_comparision_analysis.py 1" C-m

# Create new windows for folds 2-5 and run them in parallel
for fold in {2..5}; do
  tmux new-window -t adni_analysis:$((fold-1)) -n "fold-$fold"
  tmux send-keys -t adni_analysis:$((fold-1)) "cd /home2/anirudhkaushik/SYNBADD" C-m
  tmux send-keys -t adni_analysis:$((fold-1)) "conda activate pytorch" C-m
  tmux send-keys -t adni_analysis:$((fold-1)) "python adni_baseline_comparision_analysis.py $fold" C-m
done

# Attach to the tmux session
tmux attach-session -t adni_analysis
