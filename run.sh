#!/bin/bash

# echo "python main.py --configs $config --num_workers 0 --devices $CUDA_VISIBLE_DEVICES"
# CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python main.py --configs $config --num_workers 8 --devices $CUDA_VISIBLE_DEVICES

for data in NCI1 NCI109 DD PROTEINS Mutagenicity
do
	for method in NUCLEUS TOPK TAILFREE
	do
		for seed in 777 778 779 780 781 782 783 784 785 786
		do
			for ratio in 0.5 0.8
			do
				python main.py --dataset $data --seed $seed --sampling_method $method --pooling_ratio $ratio
			done
		done
	done
done

