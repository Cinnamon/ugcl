# METHOD=bare
# METHOD=joint

# METHOD=ewc
# METHOD=mas
# METHOD=gem
# METHOD=twp
# METHOD=lwf

METHOD=ergnn
# METHOD=erreplace
# METHOD=dce
# METHOD=sl
# METHOD=our

BACKBONE=GCN
# BACKBONE=GAT
# BACKBONE=GIN

CUDA_VISIBLE_DEVICES=3 python train.py \
--dataset Arxiv-CL \
--method $METHOD  \
--backbone $BACKBONE \
--ILmode classIL \
--inter-task-edges 'False' \
--minibatch 'False' \
--epochs 100 \
--ori_data_path data