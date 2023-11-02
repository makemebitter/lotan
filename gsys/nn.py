import torch
from sklearn.metrics import f1_score
import numpy as np
from sklearn.metrics import multilabel_confusion_matrix
TORCH_DTYPE = torch.float32



def unpack_conf_mat(conf_mat):
    tn, fp, fn, tp = conf_mat.ravel()
    return tn, fp, fn, tp

def f1_micro_from_cm(cm):
    tn, fp, fn, tp = unpack_conf_mat(cm)
    return tp / (tp + 1 / 2 * (fp + fn))


def cal_cm(y_pred, y_true):
    conf_mat = multilabel_confusion_matrix(
        y_true, y_pred)
    return conf_mat, y_pred.shape[0]

def f1(y_pred, y_true, multilabel=True):
    # y_true = y_true.cpu().numpy()
    # y_pred = y_pred.cpu().numpy()
    if multilabel:
        y_pred[y_pred > 0.5] = 1.0
        y_pred[y_pred <= 0.5] = 0.0
    else:
        y_pred = np.argmax(y_pred, axis=1)
    return f1_score(y_true, y_pred, average="micro"), \
        f1_score(y_true, y_pred, average="macro")


def accuracy(output, target, topk=(1, ), binary=False, return_raw=False):
    """Computes the precision@k for the specified values of k"""
    if binary:
        batch_size = target.size(0)
        _, pred = torch.max(output.data, 1)
        correct = (pred == target).sum().item()
        res = [torch.tensor(correct / batch_size)]
    else:
        maxk = max(topk)
        maxk = min(maxk, output.shape[1])
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0)
            if return_raw:
                res.append((correct_k.item(), batch_size))
            else:
                res.append(correct_k.mul_(1.0 / batch_size))
    return res


class MiniBatchLayer(torch.nn.Module):
    def apply_edge(self, H_v=None, H_u=None, X_e=None):
        raise NotImplementedError

    def aggregation(self, batch_indices=None, V=None, U=None, H_u=None):
        raise NotImplementedError

    def apply_vertex(self, X_v=None, Gamma_v=None):
        raise NotImplementedError

    def get_layer_class(self):
        raise NotImplementedError

    def forward(self, H_v, H_u, X_e, batch_indices, V, U, X_v):
        H_u_apply_eddge = self.apply_edge(H_v, H_u, X_e)
        Gamma_v = self.aggregation(
            batch_indices=batch_indices,
            V=V,
            U=U,
            H_u=H_u_apply_eddge)

        return self.apply_vertex(X_v=X_v, Gamma_v=Gamma_v)


class GCNLayer(MiniBatchLayer):

    def __init__(self,
                 input_dim,
                 output_dim,
                 relu=True,
                 dropout=None, batchnorm=False, leaky=True, xavier=False, device="cpu"):
        super(GCNLayer, self).__init__()
        layers = [torch.nn.Linear(input_dim, output_dim)]
        if relu is True:
            if leaky:
                layers.append(torch.nn.LeakyReLU())
            else:
                layers.append(torch.nn.ReLU())
        if xavier:
            torch.nn.init.xavier_uniform_(layers[0].weight)
            torch.nn.init.zeros_(layers[0].bias)
        self.model_apply_vertex = torch.nn.Sequential(*layers)
        self.add_misc_layers(dropout, batchnorm, output_dim)
        self.device = device

    def add_misc_layers(self, dropout, batchnorm, output_dim):
        self.dropout = dropout
        self.batchnorm = batchnorm
        if dropout:
            self.dropout_layer = torch.nn.Dropout(p=dropout)
        if batchnorm:
            self.batchnorm_layer = torch.nn.BatchNorm1d(output_dim)

    def apply_edge(self, H_v=None, H_u=None, X_e=None):
        return H_u

    def aggregation(self, batch_indices=None, V=None, U=None, H_u=None):
        key_val = {
            key: val
            for key, val in zip(batch_indices, range(len(batch_indices)))
        }
        latent_unique_labels = torch.tensor(
            range(len(batch_indices))).to(self.device)
        latent_labels = torch.LongTensor(
            list(map(key_val.get, V))).to(self.device)
        latent_labels = latent_labels.view(latent_labels.size(0),
                                           1).expand(-1, H_u.size(1))
        latent_unique_labels = latent_unique_labels.view(
            latent_unique_labels.size(0), 1).expand(-1, H_u.size(1))
        result = torch.zeros_like(latent_unique_labels,
                                  dtype=TORCH_DTYPE).to(self.device).scatter_add_(
                                      0, latent_labels, H_u)
        return result

    def apply_vertex(self, X_v=None, Gamma_v=None):
        act = self.model_apply_vertex(Gamma_v)
        if self.dropout:
            act = self.dropout_layer(act)
        if self.batchnorm:
            act = self.batchnorm_layer(act)
        return act


class GCNMLPLayer(GCNLayer):
    def __init__(self,
                 input_dim,
                 output_dim,
                 relu=True,
                 dropout=None,
                 batchnorm=False, leaky=True, xavier=False, mlp_hidden=[]):
        super(GCNLayer, self).__init__()
        layers = []
        if mlp_hidden:
            last = input_dim
            # overwrites behavior
            total_dims = mlp_hidden + [output_dim]
            for i, odm in enumerate(total_dims):
                layers.append(torch.nn.Linear(last, odm))
                if xavier:
                    torch.nn.init.xavier_uniform_(layers[-1].weight)
                    torch.nn.init.zeros_(layers[-1].bias)
                if i != len(total_dims) - 1:
                    # there's no point not using non-linear in mlp
                    if leaky:
                        layers.append(torch.nn.LeakyReLU())
                    else:
                        layers.append(torch.nn.ReLU())
                last = odm
            if relu is True:
                if leaky:
                    layers.append(torch.nn.LeakyReLU())
                else:
                    layers.append(torch.nn.ReLU())
            print("GIN: {}".format(layers))
        else:
            raise NotImplementedError
        self.model_apply_vertex = torch.nn.Sequential(*layers)
        self.add_misc_layers(dropout, batchnorm, output_dim)


class GCNLayerAggPushDown(GCNLayer):
    def aggregation(self, batch_indices=None, V=None, U=None, H_u=None):
        return H_u


class GCNMLPLayerAggPushDown(GCNMLPLayer):
    def aggregation(self, batch_indices=None, V=None, U=None, H_u=None):
        return H_u


class GCNLayerAggPushDownVerticalBatched(MiniBatchLayer):
    def get_layer_class(self):
        return GCNLayerAggPushDown

    def __init__(self, all_args_list, first_layer=False):

        super(GCNLayerAggPushDownVerticalBatched, self).__init__()
        self.first_layer = first_layer
        self.models = []
        self.indices = []
        self.indices_output = []
        i = 0
        j = 0
        for args, kwargs in all_args_list:
            model = self.get_layer_class()(*args, **kwargs)
            input_dim, output_dim = args[0], args[1]
            self.indices.append((i, i + input_dim))
            self.indices_output.append((j, j + output_dim))
            self.models.append(model)
            if not self.first_layer:
                # first layer share inputs
                i += input_dim
            j += output_dim
        self.models = torch.nn.ModuleList(self.models)
        self.input_dim = i
        self.output_dim = j
        print(self.indices, self.indices_output)

    def forward(self, H_v, H_u, X_e, batch_indices, V, U, X_v):
        act = []
        for (begin, end), model in zip(self.indices, self.models):
            # print((begin, end), H_u.shape)
            H_u_submodel = H_u[:, begin:end]
            x = model(H_v, H_u_submodel, X_e, batch_indices, V, U, X_v)
            act.append(x)

        act = torch.cat(act, dim=1)
        return act

        # H_u_apply_eddge = self.apply_edge(H_v, H_u, X_e)
        # Gamma_v = self.aggregation(
        #     batch_indices=batch_indices,
        #     V=V,
        #     U=U,
        #     H_u=H_u_apply_eddge)

        # return self.apply_vertex(X_v=X_v, Gamma_v=Gamma_v)

    # def aggregation(self, batch_indices=None, V=None, U=None, H_u=None):
    #     return H_u

    # def apply_edge(self, H_v=None, H_u=None, X_e=None):
    #     return H_u

    # def apply_vertex(self, X_v=None, Gamma_v=None):

    #     pass

    # def eval(self):
    #     for model in self.models:
    #         model.eval()

    # def train(self):
    #     for model in self.models:
    #         model.train

    # def no_sync(self):
    #     for model in self.models:
    #         model.no


class GINLayerAggPushDownVerticalBatched(GCNLayerAggPushDownVerticalBatched):
    def get_layer_class(self):
        return GCNMLPLayerAggPushDown
