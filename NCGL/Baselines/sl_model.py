import torch
import copy
from .ergnn_utils import *
import pickle
from dgl.utils import expand_as_pair

samplers = {'CM': CM_sampler(plus=False), 'CM_plus':CM_sampler(plus=True), 'MF':MF_sampler(plus=False), 'MF_plus':MF_sampler(plus=True),'random':random_sampler(plus=False)}
K_SAMPLES = 10


class NET(torch.nn.Module):

    """
    ER-GNN baseline for NCGL tasks

    :param model: The backbone GNNs, e.g. GCN, GAT, GIN, etc.
    :param task_manager: Mainly serves to store the indices of the output dimensions corresponding to each task
    :param args: The arguments containing the configurations of the experiments including the training parameters like the learning rate, the setting confugurations like class-IL and task-IL, etc. These arguments are initialized in the train.py file and can be specified by the users upon running the code.

    """

    def __init__(self,
                 model,
                 task_manager,
                 args):
        super(NET, self).__init__()

        self.task_manager = task_manager

        # setup network
        self.net = model
        self.sampler = samplers[args.sl_args['sampler']]

        # setup optimizer
        self.opt = torch.optim.Adam(self.net.parameters(), lr=args.lr, weight_decay=args.weight_decay)

        # setup losses
        self.ce = torch.nn.functional.cross_entropy

        # setup memories
        self.current_task = -1
        self.buffer_node_ids = []
        self.budget = int(args.sl_args['budget'])
        self.max_size = int(args.sl_args['max_size'] * args.n_cls * self.budget)
        self.d_CM = args.sl_args['d'] # d for CM sampler of ERGNN
        self.aux_g = None

    def forward(self, features):
        output = self.net(features)
        return output

    def observe(self, args, g, features, labels, t, prev_model, train_ids, ids_per_cls, dataset):
        """
        The method for learning the given tasks under the class-IL setting.

        :param args: Same as the args in __init__().
        :param g: The graph of the current task.
        :param features: Node features of the current task.
        :param labels: Labels of the nodes in the current task.
        :param t: Index of the current task.
        :prev_model: The previous task model.
        :param train_ids: The indices of the nodes participating in the training.
        :param ids_per_cls: Indices of the nodes in each class.
        :param dataset: The entire dataset.

        """
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        ids_per_cls_train = [list(set(ids).intersection(set(train_ids))) for ids in ids_per_cls]
        self.net.train()
        n_nodes = len(train_ids)
        buffer_size = len(self.buffer_node_ids)
        beta = buffer_size/(buffer_size+n_nodes)

        self.net.zero_grad()
        offset1, offset2 = self.task_manager.get_label_offset(t)
        output, _ = self.net(g, features)
        output_labels = labels[train_ids]

        if args.cls_balance:
            n_per_cls = [(output_labels == j).sum() for j in range(args.n_cls)]
            loss_w_ = [1. / max(i, 1) for i in n_per_cls]  # weight to balance the loss of different class
        else:
            loss_w_ = [1. for i in range(args.n_cls)]
        loss_w_ = torch.tensor(loss_w_).to(device=device)
        if args.classifier_increase:
            loss = self.ce(output[train_ids,offset1:offset2], labels[train_ids], weight=loss_w_[offset1: offset2])
        else:
            loss = self.ce(output[train_ids], labels[train_ids], weight=loss_w_)

        # sample and store ids from current task
        # store only once for each task
        if t!=self.current_task:
            self.current_task = t
            sampled_ids = self.sampler(ids_per_cls_train, self.budget, features, self.net.second_last_h.detach(), self.d_CM)
            old_ids = g.ndata['_ID'].cpu() # '_ID' are the original ids in the original graph before splitting
            self.buffer_node_ids.extend(old_ids[sampled_ids].tolist())
            if t>0:
                g, __, _ = dataset.get_graph(node_ids=self.buffer_node_ids)
                self.aux_g = g.to(device=device)
                self.aux_features, self.aux_labels = self.aux_g.srcdata['feat'], self.aux_g.dstdata['label'].squeeze()
                # TODO: when self.buffer_node_ids is full, start to replace nodes with more class
                if len(self.buffer_node_ids) > self.max_size:
                    print(f"Current size of replay buffer {len(self.buffer_node_ids)} > max_size")
                    _ids_per_cls = [torch.nonzero(self.aux_labels == j).squeeze().tolist() for j in range(args.n_cls)]
                    _node_ids_per_cls = [[self.buffer_node_ids[idx] for idx in ids] for ids in _ids_per_cls]
                    while len(self.buffer_node_ids) > self.max_size:
                        largest_cls = max(enumerate(_node_ids_per_cls), key=lambda item: len(item[1]))[0]
                        _removed = random.choice(_node_ids_per_cls[largest_cls])
                        _node_ids_per_cls[largest_cls].remove(_removed)
                        self.buffer_node_ids.remove(_removed)

                # TODO: run it again with the correct size
                g, __, _ = dataset.get_graph(node_ids=self.buffer_node_ids)
                self.aux_g = g.to(device=device)
                self.aux_features, self.aux_labels = self.aux_g.srcdata['feat'], self.aux_g.dstdata['label'].squeeze()

                if args.cls_balance:
                    n_per_cls = [(self.aux_labels == j).sum() for j in range(args.n_cls)]
                    loss_w_ = [1. / max(i, 1) for i in n_per_cls]  # weight to balance the loss of different class
                else:
                    loss_w_ = [1. for i in range(args.n_cls)]
                self.aux_loss_w_ = torch.tensor(loss_w_).to(device=device)

        if t!=0:
            # calculate auxiliary loss based on replay if not the first task
            output, _ = self.net(self.aux_g, self.aux_features)
            if args.classifier_increase:
                loss_aux = self.ce(output[:, offset1:offset2], self.aux_labels, weight=self.aux_loss_w_[offset1: offset2])
            else:
                loss_aux = self.ce(output, self.aux_labels, weight=self.aux_loss_w_)

            structure_loss = 0
            if prev_model is not None:
                # If there is a previous model, then we get the previous model's logits to calculate the distillation loss.
                prev_output, _ = prev_model(self.aux_g, self.aux_features)
                # adj_matrix = self.aux_g.adj()
                feat_src, _ = expand_as_pair(self.aux_features)
                self.aux_g.srcdata['h'] = feat_src
                self.aux_g.apply_edges(lambda edges: {'se': torch.sum((torch.mul(edges.src['h'], torch.tanh(edges.dst['h']))), 1)})
                soft_edges = self.aux_g.edata.pop('se')
                rand_k_node_samples = random.sample(range(0, self.aux_g.num_nodes()), K_SAMPLES)

                for node_idx in rand_k_node_samples:
                    # For the old (previous task) model.
                    # Get the different in term of features between the target node and its neighbor nodes. (This aims to extract the
                    # structure information between the node and its neighbors).
                    prev_feats = prev_output[:, offset1:offset2]
                    # ref_neighbor_nodes = prev_feats[adj_matrix[node_idx].to_dense().bool()]
                    ref_neighbor_nodes = soft_edges.unsqueeze(1) * prev_feats
                    if ref_neighbor_nodes.numel() > 0:
                        ref_neighbors_feat = ref_neighbor_nodes.sum(dim=0)
                        ref_diff_vector = prev_feats[node_idx] - ref_neighbors_feat
                    else:
                        ref_diff_vector = None

                    # For the current model.
                    # Get the different in term of features between the target node and its neighbor nodes. (This aims to extract the
                    # structure information between the node and its neighbors).
                    cur_feats = output[:, offset1:offset2]
                    # cur_neighbor_nodes = cur_feats[adj_matrix[node_idx].to_dense().bool()]
                    cur_neighbor_nodes = soft_edges.unsqueeze(1) * cur_feats
                    if cur_neighbor_nodes.numel() > 0:
                        cur_neighbors_feat = cur_neighbor_nodes.sum(dim=0)
                        cur_diff_vector = cur_feats[node_idx] - cur_neighbors_feat
                    else:
                        cur_diff_vector = None

                    if ref_diff_vector is not None and cur_diff_vector is not None:
                        if (ref_diff_vector == cur_diff_vector).all():
                            # Skip if two vectors are similar.
                            continue
                        
                        # Calculate the difference (similarity) of the learned structure information between the old model and the
                        # current model.
                        step_structure_loss = nn.CosineEmbeddingLoss()(torch.unsqueeze(ref_diff_vector, dim=0),
                                                                        torch.unsqueeze(cur_diff_vector, dim=0),
                                                                        torch.ones(1).to(device=device))
                        structure_loss += step_structure_loss

            loss = beta * loss + (1 - beta) * (loss_aux + structure_loss)

        loss.backward()
        self.opt.step()

    def observe_task_IL(self, args, g, features, labels, t, prev_model, train_ids, ids_per_cls, dataset):
        """
        The method for learning the given tasks under the task-IL setting.

        :param args: Same as the args in __init__().
        :param g: The graph of the current task.
        :param features: Node features of the current task.
        :param labels: Labels of the nodes in the current task.
        :param t: Index of the current task.
        :param train_ids: The indices of the nodes participating in the training.
        :param ids_per_cls: Indices of the nodes in each class.
        :param dataset: The entire dataset.

        """
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        ids_per_cls_train = [list(set(ids).intersection(set(train_ids))) for ids in ids_per_cls]
        if not isinstance(self.aux_g, list):
            self.aux_g = []
            self.buffer_node_ids = {}
            self.aux_loss_w_ = []
        self.net.train()
        n_nodes = len(train_ids)
        buffer_size = 0
        for k in self.buffer_node_ids:
            buffer_size+=len(self.buffer_node_ids[k])
        beta = buffer_size/(buffer_size+n_nodes)

        self.net.zero_grad()
        offset1, offset2 = self.task_manager.get_label_offset(t - 1)[1], self.task_manager.get_label_offset(t)[1]
        output, _ = self.net(g, features)
        output_labels = labels[train_ids]

        if args.cls_balance:
            n_per_cls = [(output_labels == j).sum() for j in range(args.n_cls)]
            loss_w_ = [1. / max(i, 1) for i in n_per_cls]  # weight to balance the loss of different class
        else:
            loss_w_ = [1. for i in range(args.n_cls)]
        loss_w_ = torch.tensor(loss_w_).to(device=device)
        loss = self.ce(output[train_ids, offset1:offset2], output_labels-offset1, weight=loss_w_[offset1: offset2])

        if t!=self.current_task:
            self.current_task = t
            sampled_ids = self.sampler(ids_per_cls_train, self.budget, features, self.net.second_last_h.detach(), self.d_CM)
            old_ids = g.ndata['_ID'].cpu()
            self.buffer_node_ids[t] = old_ids[sampled_ids].tolist()
            g, __, _ = dataset.get_graph(node_ids=self.buffer_node_ids[t])
            self.aux_g.append(g.to(device=device))
            if args.cls_balance:
                n_per_cls = [(labels[sampled_ids] == j).sum() for j in range(args.n_cls)]
                loss_w_ = [1. / max(i, 1) for i in n_per_cls]  # weight to balance the loss of different class
            else:
                loss_w_ = [1. for i in range(args.n_cls)]
            loss_w_ = torch.tensor(loss_w_).to(device=device)
            self.aux_loss_w_.append(loss_w_)

        if t!=0:
            for oldt in range(t):
                o1, o2 = self.task_manager.get_label_offset(oldt - 1)[1], self.task_manager.get_label_offset(oldt)[1]
                aux_g = self.aux_g[oldt]
                aux_features, aux_labels = aux_g.srcdata['feat'], aux_g.dstdata['label'].squeeze()
                output, _ = self.net(aux_g, aux_features)
                loss_aux = self.ce(output[:, o1:o2], aux_labels - o1, weight=self.aux_loss_w_[oldt][o1: o2])
                loss = beta * loss + (1 - beta) * loss_aux

        loss.backward()
        self.opt.step()

    def observe_task_IL_batch(self, args, g, dataloader, features, labels, t, prev_model, train_ids, ids_per_cls, dataset):
        """
        The method for learning the given tasks under the task-IL setting with mini-batch training.

        :param args: Same as the args in __init__().
        :param g: The graph of the current task.
        :param dataloader: The data loader for mini-batch training
        :param features: Node features of the current task.
        :param labels: Labels of the nodes in the current task.
        :param t: Index of the current task.
        :param train_ids: The indices of the nodes participating in the training.
        :param ids_per_cls: Indices of the nodes in each class (currently not in use).
        :param dataset: The entire dataset (currently not in use).

        """
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        ids_per_cls_train = [list(set(ids).intersection(set(train_ids))) for ids in ids_per_cls]
        if not isinstance(self.aux_g, list):
            self.aux_g = []
            self.buffer_node_ids = {}
            self.aux_loss_w_ = []
        self.net.train()
        # now compute the grad on the current task
        offset1, offset2 = self.task_manager.get_label_offset(t-1)[1], self.task_manager.get_label_offset(t)[1]
        for input_nodes, output_nodes, blocks in dataloader:
            n_nodes_current_batch = output_nodes.shape[0]
            buffer_size = 0
            for k in self.buffer_node_ids:
                buffer_size += len(self.buffer_node_ids[k])
            beta = buffer_size / (buffer_size + n_nodes_current_batch)
            self.net.zero_grad()
            blocks = [b.to(device=device) for b in blocks]
            input_features = blocks[0].srcdata['feat']
            output_labels = blocks[-1].dstdata['label'].squeeze()

            if args.cls_balance:
                n_per_cls = [(output_labels == j).sum() for j in range(args.n_cls)]
                loss_w_ = [1. / max(i, 1) for i in n_per_cls]  # weight to balance the loss of different class
            else:
                loss_w_ = [1. for i in range(args.n_cls)]
            loss_w_ = torch.tensor(loss_w_).to(device=device)
            output_labels = output_labels - offset1
            output_predictions,_ = self.net.forward_batch(blocks, input_features)
            loss = self.ce(output_predictions[:, offset1:offset2], output_labels, weight=loss_w_[offset1: offset2])

            # sample and store ids from current task
            if t != self.current_task:
                self.current_task = t
                sampled_ids = self.sampler(ids_per_cls_train, self.budget, features.to(device=device), self.net.second_last_h.detach(), self.d_CM)
                old_ids = g.ndata['_ID'].cpu()
                self.buffer_node_ids[t] = old_ids[sampled_ids].tolist()
                ag, __, _ = dataset.get_graph(node_ids=self.buffer_node_ids[t])
                self.aux_g.append(ag.to(device=device))
                if args.cls_balance:
                    n_per_cls = [(labels[sampled_ids] == j).sum() for j in range(args.n_cls)]
                    loss_w_ = [1. / max(i, 1) for i in n_per_cls]  # weight to balance the loss of different class
                else:
                    loss_w_ = [1. for i in range(args.n_cls)]
                loss_w_ = torch.tensor(loss_w_).to(device=device)
                self.aux_loss_w_.append(loss_w_)

            if t != 0:
                for oldt in range(t):
                    o1, o2 = self.task_manager.get_label_offset(oldt-1)[1], self.task_manager.get_label_offset(oldt)[1]
                    aux_g = self.aux_g[oldt]
                    aux_features, aux_labels = aux_g.srcdata['feat'], aux_g.dstdata['label'].squeeze()
                    output, _ = self.net(aux_g, aux_features)
                    loss_aux = self.ce(output[:, o1:o2], aux_labels - o1, weight=self.aux_loss_w_[oldt][o1:o2])
                    loss = beta * loss + (1 - beta) * loss_aux
            loss.backward()
            self.opt.step()

    def observe_class_IL_batch(self, args, g, dataloader, features, labels, t, prev_model, train_ids, ids_per_cls, dataset):
        """
        The method for learning the given tasks under the class-IL setting with mini-batch training.

        :param args: Same as the args in __init__().
        :param g: The graph of the current task.
        :param dataloader: The data loader for mini-batch training
        :param features: Node features of the current task.
        :param labels: Labels of the nodes in the current task.
        :param t: Index of the current task.
        :param train_ids: The indices of the nodes participating in the training.
        :param ids_per_cls: Indices of the nodes in each class (currently not in use).
        :param dataset: The entire dataset (currently not in use).

        """
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        ids_per_cls_train = [list(set(ids).intersection(set(train_ids))) for ids in ids_per_cls]
        self.net.train()
        # now compute the grad on the current task
        offset1, offset2 = self.task_manager.get_label_offset(t)
        for input_nodes, output_nodes, blocks in dataloader:
            n_nodes_current_batch = output_nodes.shape[0]
            buffer_size = len(self.buffer_node_ids)
            beta = buffer_size / (buffer_size + n_nodes_current_batch)
            self.net.zero_grad()
            blocks = [b.to(device=device) for b in blocks]
            input_features = blocks[0].srcdata['feat']
            output_labels = blocks[-1].dstdata['label'].squeeze()

            if args.cls_balance:
                n_per_cls = [(output_labels == j).sum() for j in range(args.n_cls)]
                loss_w_ = [1. / max(i, 1) for i in n_per_cls]  # weight to balance the loss of different class
            else:
                loss_w_ = [1. for i in range(args.n_cls)]
            loss_w_ = torch.tensor(loss_w_).to(device=device)
            output_predictions,_ = self.net.forward_batch(blocks, input_features)
            loss = self.ce(output_predictions[:, offset1:offset2], output_labels, weight=loss_w_[offset1: offset2])

            # sample and store ids from current task
            # store only once for each task
            if t != self.current_task:
                self.current_task = t
                sampled_ids = self.sampler(ids_per_cls_train, self.budget, features.to(device=device), self.net.second_last_h.detach(), self.d_CM)
                old_ids = g.ndata['_ID'].cpu()
                self.buffer_node_ids.extend(old_ids[sampled_ids].tolist())
                if t > 0:
                    g, __, _ = dataset.get_graph(node_ids=self.buffer_node_ids)
                    self.aux_g = g.to(device=device)
                    self.aux_features, self.aux_labels = self.aux_g.srcdata['feat'], self.aux_g.dstdata['label'].squeeze()

                    # TODO: when self.buffer_node_ids is full, start to replace nodes with more class
                    if len(self.buffer_node_ids) > self.max_size:
                        print(f"Current size of replay buffer {len(self.buffer_node_ids)} > max_size")
                        _ids_per_cls = [torch.nonzero(self.aux_labels == j).squeeze().tolist() for j in range(args.n_cls)]
                        _node_ids_per_cls = [[self.buffer_node_ids[idx] for idx in ids] for ids in _ids_per_cls]
                        while len(self.buffer_node_ids) > self.max_size:
                            largest_cls = max(enumerate(_node_ids_per_cls), key=lambda item: len(item[1]))[0]
                            _removed = random.choice(_node_ids_per_cls[largest_cls])
                            _node_ids_per_cls[largest_cls].remove(_removed)
                            self.buffer_node_ids.remove(_removed)

                    # TODO: run it again with the correct size
                    g, __, _ = dataset.get_graph(node_ids=self.buffer_node_ids)
                    self.aux_g = g.to(device=device)
                    self.aux_features, self.aux_labels = self.aux_g.srcdata['feat'], self.aux_g.dstdata['label'].squeeze()

                    if args.cls_balance:
                        n_per_cls = [(self.aux_labels == j).sum() for j in range(args.n_cls)]
                        loss_w_ = [1. / max(i, 1) for i in n_per_cls]  # weight to balance the loss of different class
                    else:
                        loss_w_ = [1. for i in range(args.n_cls)]
                    self.aux_loss_w_ = torch.tensor(loss_w_).to(device=device)

            if t != 0:
                output, _ = self.net(self.aux_g, self.aux_features)
                if args.classifier_increase:
                    loss_aux = self.ce(output[:, offset1:offset2], self.aux_labels,
                                       weight=self.aux_loss_w_[offset1: offset2])
                else:
                    loss_aux = self.ce(output, self.aux_labels, weight=self.aux_loss_w_)

                structure_loss = 0
                if prev_model is not None:
                    # If there is a previous model, then we get the previous model's logits to calculate the distillation loss.
                    prev_output, edge_list = prev_model(self.aux_g, self.aux_features)
                    # # adj_matrix = self.aux_g.adj()
                    # src, dst = self.aux_g.edges()
                    # adj_matrix = torch.zeros(self.aux_g.num_src_nodes(), self.aux_g.num_dst_nodes(), device=device)
                    # adj_matrix[src, dst] = self.aux_g.edata['_ID'].squeeze(-1).float()

                    feat_src, _ = expand_as_pair(self.aux_features)
                    self.aux_g.srcdata['h'] = feat_src
                    self.aux_g.apply_edges(lambda edges: {'se': torch.sum((torch.mul(edges.src['h'], torch.tanh(edges.dst['h']))), 1)})
                    soft_edges = self.aux_g.edata.pop('se')

                    rand_k_node_samples = random.sample(range(0, self.aux_g.num_nodes()), K_SAMPLES)
                    for node_idx in rand_k_node_samples:
                        # For the old (previous task) model.
                        # Get the different in term of features between the target node and its neighbor nodes. (This aims to extract the
                        # structure information between the node and its neighbors).
                        prev_feats = prev_output[:, offset1:offset2]
                        # ref_neighbor_nodes = prev_feats[adj_matrix[node_idx].to_dense().bool()]
                        ref_neighbor_nodes = soft_edges.unsqueeze(1) * prev_feats
                        if ref_neighbor_nodes.numel() > 0:
                            ref_neighbors_feat = ref_neighbor_nodes.sum(dim=0)
                            ref_diff_vector = prev_feats[node_idx] - ref_neighbors_feat
                        else:
                            ref_diff_vector = None

                        # For the current model.
                        # Get the different in term of features between the target node and its neighbor nodes. (This aims to extract the
                        # structure information between the node and its neighbors).
                        cur_feats = output[:, offset1:offset2]
                        # cur_neighbor_nodes = cur_feats[adj_matrix[node_idx].to_dense().bool()]
                        cur_neighbor_nodes = soft_edges.unsqueeze(1) * cur_feats
                        if cur_neighbor_nodes.numel() > 0:
                            cur_neighbors_feat = cur_neighbor_nodes.sum(dim=0)
                            cur_diff_vector = cur_feats[node_idx] - cur_neighbors_feat
                        else:
                            cur_diff_vector = None

                        if ref_diff_vector is not None and cur_diff_vector is not None:
                            if (ref_diff_vector == cur_diff_vector).all():
                                # Skip if two vectors are similar.
                                continue
                            
                            # Calculate the difference (similarity) of the learned structure information between the old model and the
                            # current model.
                            step_structure_loss = nn.CosineEmbeddingLoss()(torch.unsqueeze(ref_diff_vector, dim=0),
                                                                           torch.unsqueeze(cur_diff_vector, dim=0),
                                                                           torch.ones(1).to(device=device))
                            structure_loss += step_structure_loss

                loss = beta * loss + (1 - beta) * (loss_aux + structure_loss)

            loss.backward()
            self.opt.step()
