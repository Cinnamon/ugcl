# METHOD=bare
# METHOD=independent
METHOD=jointtrain
# METHOD=gem
# METHOD=ergnn
# METHOD=erreplace
# METHOD=dce
# METHOD=sl
# METHOD=our

CUDA_VISIBLE_DEVICES=3 python GCGL/train.py \
	--dataset ENZYMES-CL \
	--method $METHOD \
	--backbone GCN \
	
	--clsIL 'True' \
	--num_epochs 100 \
	--result_path GCGL/results \
	--overwrite_result 'False'

