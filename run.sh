#!/bin/bash

# echo "python main.py --configs $config --num_workers 0 --devices $CUDA_VISIBLE_DEVICES"
# CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python main.py --configs $config --num_workers 8 --devices $CUDA_VISIBLE_DEVICES

for data in IMDB-BINARY IMDB-MULTI COLLAB # NCI1 NCI109 DD PROTEINS Mutagenicity MUTAG ENZYMES
do
	for method in ITS TOPK NUCLEUS TAILFREE  
	do
		for seed in 777 778 779 780 781 782 783 784 785 786
		do
			if [ "$method" = "NUCLEUS" ]
			then python main.py --dataset $data --seed $seed --sampling_method NUCLEUS --pooling_ratio 0.8
			fi
			if [ "$method" = "TOPK" ]
			then python main.py --dataset $data --seed $seed --sampling_method TOPK --pooling_ratio 0.5
			fi
			if [ "$method" = "TAILFREE" ]
			then python main.py --dataset $data --seed $seed --sampling_method TAILFREE --pooling_ratio 0.8
			fi
			if [ "$method" = "ITS" ]
			then
			for ratio in 0.5 0.6
			do
				python main.py --dataset $data --seed $seed --sampling_method ITS --pooling_ratio $ratio
			done
			fi
		done
	done
done

