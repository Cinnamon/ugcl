METHOD=ergnn
# METHOD=erreplace
# METHOD=dce
# METHOD=sl
# METHOD=our

CUDA_VISIBLE_DEVICES=3 python train.py \
--dataset Reddit-CL \
--method $METHOD  
--ILmode classIL \
--inter-task-edges 'False' \
--minibatch 'True' \
--epochs 100 \
--ori_data_path data