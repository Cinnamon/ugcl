METHOD=ergnn
# METHOD=erreplace
# METHOD=dce
# METHOD=sl
# METHOD=our

BACKBONE=GAT

CUDA_VISIBLE_DEVICES=2 python train.py \
--dataset CoraFull-CL \
--method $METHOD  \
--backbone $BACKBONE

--ILmode classIL \
--inter-task-edges 'False' \
--minibatch 'False' \
--epochs 100 \
--ori_data_path data