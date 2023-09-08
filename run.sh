#!/bin/bash

# echo "python main.py --configs $config --num_workers 0 --devices $CUDA_VISIBLE_DEVICES"
# CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python main.py --configs $config --num_workers 8 --devices $CUDA_VISIBLE_DEVICES



for method in NUCLEUS TOPK
do
	for seed in 777 778 779 780 781 782 783 784 785 786
	do
		python main.py --seed $seed --sampling_method $method
	done
done

