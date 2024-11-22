import torch
from torch.optim import Adam
from dgllife.utils import Meter
def predict(args, model, bg, device=None):
    node_feats = bg.ndata[args['node_data_field']].to(device=device)
    if args.get('edge_featurizer', None) is not None:
        edge_feats = bg.edata[args['edge_data_field']].to(device=device)
        return model(bg, node_feats, edge_feats)

    return model(bg, node_feats)

class NET(torch.nn.Module):
    """
    Jointly trained model baseline for GCGL tasks

    :param model: The backbone GNNs, e.g. GCN, GAT, GIN, etc.
    :param args: The arguments containing the configurations of the experiments including the training parameters like the learning rate, the setting confugurations like class-IL and task-IL, etc. These arguments are initialized in the train.py file and can be specified by the users upon running the code.

    """

    def __init__(self,
                 model,
                 args):
        super(NET, self).__init__()

        # setup network
        self.net = model
        self.optimizer = Adam(model.parameters(), lr=args['lr'])

    def forward(self, features):
        output = self.net(features)
        return output

    def observe(self, data_loader, loss_criterion, task_i, args, device=None):
        """
        The method for learning the given tasks under the task-IL setting with multi-label classification datasets.

        :param data_loader: The data loader for mini-batch training.
        :param loss_criterion: The loss function.
        :param task_i: Index of the current task.
        :param args: Same as the args in __init__().

        """

        self.net.train()
        train_meter = Meter()
        for batch_id, batch_data in enumerate(data_loader):
            smiles, bg, labels, masks = batch_data
            bg = bg.to(device=device)
            labels, masks = labels.to(device=device), masks.to(device=device)
            logits = predict(args, self.net, bg, task_i)

            # Mask non-existing labels
            loss_all = loss_criterion(logits, labels) * (masks != 0).float()
            loss = loss_all[:,task_i].mean()

            for old_t in range(task_i):
                loss = loss + loss_all[:,old_t].mean()
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            train_meter.update(logits, labels, masks)


    def observe_tskIL_multicls(self, data_loader, loss_criterion, task_i, args, device=None):
        """
        The method for learning the given tasks under the task-IL setting with multi-class classification datasets.

        :param data_loader: The data loader for mini-batch training.
        :param loss_criterion: The loss function.
        :param task_i: Index of the current task.
        :param args: Same as the args in __init__().

        """
        # task Il under multi-class setting
        self.net.train()
        loss = 0
        for oldt_id, old_t in enumerate(args['tasks'][0:task_i + 1]):  # range(task_i):
            for batch_id, batch_data in enumerate(data_loader[oldt_id]):
                smiles, bg, labels, masks = batch_data
                labels, masks = labels.to(device=device), masks.to(device=device)
                logits = predict(args, self.net, bg.to(device=device), oldt_id)

                # class balance
                n_per_cls = [(labels == j).sum() for j in old_t]
                loss_w_ = [1. / max(i, 1) for i in n_per_cls]
                loss_w_ = torch.tensor(loss_w_).to(device=device)
                for i, c in enumerate(old_t):
                    labels[labels == c] = i
                loss += loss_criterion(logits[:, old_t], labels.long(), weight=loss_w_).float()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def observe_clsIL(self, data_loader, loss_criterion, task_i, args, device=None):
        """
        The method for learning the given tasks under the class-IL setting with multi-class classification datasets.

        :param data_loader: The data loader for mini-batch training.
        :param loss_criterion: The loss function.
        :param task_i: Index of the current task.
        :param args: Same as the args in __init__().

        """

        self.net.train()
        clss = []
        for tid in range(task_i + 1):
            clss.extend(args['tasks'][tid])

        for batch_id, batch_data in enumerate(data_loader[task_i]):
            # for joint training, dataloader[task_i] contains data from task 0 to task_i
            smiles, bg, labels, masks = batch_data
            bg = bg.to(device=device)
            labels, masks = labels.to(device=device), masks.to(device=device)
            logits, _ = predict(args, self.net, bg)

            # class balance
            n_per_cls = [(labels == j).sum() for j in clss]
            loss_w_ = [1. / max(i, 1) for i in n_per_cls]
            loss_w_ = torch.tensor(loss_w_).to(device=device)
            for i, c in enumerate(clss):
                labels[labels == c] = i
            loss = loss_criterion(logits[:, clss], labels.long(), weight=loss_w_).float()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
