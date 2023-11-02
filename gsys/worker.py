# Copyright 2023 Yuhao Zhang and Arun Kumar. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

#  a worker that listens for Messengers,
# two processes one reads from the pipes and
# maintains a message queue,
# and the other consumes the queue and writes back to the named pipes
from .ipc import IPCBase
from .ipc import serialize
from .ipc import deserialize
from .ipc import create_shm_array
from .ipc import DatumSHM
from .constants import network_constants
from .utils import log_local_test_acc
from .messages import messages
import pickle
from queue import Empty
import numpy as np
import multiprocessing
import os
from .utils import logs
from .utils import logsc

from .nn import accuracy
from .nn import cal_cm
from .nn import f1
from .nn import f1_micro_from_cm
import torch
import copy
from collections import defaultdict
from .nn import TORCH_DTYPE
from .all_args import get_rank
import gc
DEBUG = False


def batch_data_gen(incoming_mq, batch_size):
    # should be some termination criteria here
    while True:
        # fetch batch_size number of data
        data_batch = []
        size = 0
        while size < batch_size:
            try:
                ident, msg = incoming_mq.get(
                    timeout=network_constants.INCOMING_MQ_TIMEOUT)
                obj = pickle.loads(msg)
                data_batch.append((ident, obj))
                size += 1
            except Empty:
                if size > 0:
                    break

        yield data_batch


def plain_sum(json_obj):
    vertex_id = json_obj['_1']
    map_of_embs = json_obj['_2']
    mat_embs = [x['_2'] for x in map_of_embs]
    summed_emb = np.asarray(mat_embs, dtype=np.float32).sum(axis=0).tolist()
    ret = {'_1': vertex_id, '_2': summed_emb}
    return ret


class Worker(IPCBase):
    def main(self, incoming_mq, outgoing_mq):
        #     mimick pytorch work
        bdg = batch_data_gen(incoming_mq, 5)

        # process indefinitely
        while True:
            # single-process feed to pytorch
            batch = next(bdg)
            # dummy work, do aggregation
            out_batch = []
            for ident, obj in batch:
                # print(ident, obj)
                out_batch.append((ident, plain_sum(obj)))

            for ident, obj in out_batch:
                msg = pickle.dumps(obj)
                outgoing_mq.put((ident, msg))


class PreBatchedWorker(IPCBase):
    def get_device(self):
        if torch.cuda.is_available() and self.args.gpu:
            device = torch.device("cuda:0")
            # device_ids = list(range(torch.cuda.device_count()))
            print('GPU detected')
        else:
            device = torch.device("cpu")
            print('No GPU or GPU disabled. switching to CPU')
        return device

    def serialize(self, objs):
        return serialize(objs)

    def deserialize(self, msg):
        return deserialize(msg)

    def nbgenerator(self, incoming_mq):
        while True:
            try:
                ident, msg = incoming_mq.get(
                    timeout=network_constants.INCOMING_MQ_TIMEOUT)
                datum = self.deserialize(msg)
                yield ident, datum
            except Empty:
                yield None

    def generator(self, incoming_mq):
        while True:
            ident, msg = incoming_mq.get()
            datum = self.deserialize(msg)
            yield ident, datum

    def mini_batch_packed_cast(self, result):
        with logsc(
            "{}_mini_batch_packed_list_cast".format(self.direction),
            elapsed_time=True,
                log_dict=self.ward_log, accumulate=True):
            rlist = result.cpu().numpy()

        return rlist

    def mini_batch_packed(self, batch_indices, result):
        # `result` can be on GPU or CPU
        rlist = self.mini_batch_packed_cast(result)
        return list(zip(batch_indices, rlist))

    # def mini_batch_packed_lift_json(self, batch_indices, result):
    #     json_objs = []
    #     for batch_index, payload in zip(batch_indices, result.tolist()):
    #         json_obj = {
    #             '_1': batch_index,
    #             '_2': payload
    #         }
    #         json_objs.append(json_obj)
    #     return json_objs

    def mini_batch_packed_grad(self, batch_indices, V, U, H_u_grad):
        # `H_u_grad` can be on GPU or CPU, `V` and `U`
        # would be better to on CPU
        print("CALLING mini_batch_packed_grad")
        objs = []
        objs_index = {}
        for v_idx in batch_indices:
            obj = (v_idx, [])

            objs.append(obj)
            objs_index[v_idx] = obj

        for v_idx, u_idx, payload in zip(V, U, H_u_grad.cpu().numpy()):
            obj_u = (u_idx, payload)
            objs_index[v_idx][1].append(obj_u)

        return objs

    # def mini_batch_packed_grad_lift_json(
    #   self, batch_indices, V, U, H_u_grad):
    #     json_objs = []
    #     json_objs_index = {}
    #     for v_idx in batch_indices:
    #         json_obj = {
    #             '_1': v_idx,
    #             '_2': []
    #         }
    #         json_objs.append(json_obj)
    #         json_objs_index[v_idx] = json_obj
    #     for v_idx, u_idx, payload in zip(V, U, H_u_grad.tolist()):
    #         json_obj_u = {
    #             '_1': u_idx,
    #             '_2': payload
    #         }
    #         json_objs_index[v_idx]['_2'].append(json_obj_u)

    #     return json_objs

    def dist_init(self):
        rank = get_rank()
        if torch.cuda.is_available() and self.args.gpu:
            backend = 'nccl'
        else:
            backend = 'gloo'
        logs(
            "DDP Initializing ... Rank: {}, World Size: {}".format(
                rank, self.args.size))
        torch.distributed.init_process_group(
            backend=backend,
            init_method='tcp://{}:23456'.format(self.args.master),
            rank=rank, world_size=self.args.size)

        logs("DDP Initialized: {}".format(torch.distributed.is_initialized()))

    def model_init(self):

        hidden = self.args.cats.num_hidden
        num_classes = self.args.num_classes
        num_layers = self.args.cats.num_layers

        self.models = []
        optimizer_lookup = defaultdict(list)
        params_lookup = defaultdict(list)
        for layer in range(num_layers):
            if layer == 0:
                input_dim = self.args.feature_shape
                output_dim = hidden

            elif layer == (num_layers - 1):
                # last layer, prediction
                input_dim = hidden
                output_dim = num_classes
            else:
                input_dim = hidden
                output_dim = hidden

            if self.args.lotan_model_batching:

                if self.args.model == "gin":
                    from .nn import GINLayerAggPushDownVerticalBatched as GCNBatchedLayer
                else:
                    from .nn import GCNLayerAggPushDownVerticalBatched as GCNBatchedLayer
                all_args_list = []

                for i, mst in enumerate(self.args.cats.hs_msts):
                    # fall back to default should the mst doesn't have
                    # if 'optimizer' not in mst:
                    #     mst['optimizer'] = self.args.cats.optimizer
                    # if 'lr' not in mst:
                    #     mst['lr'] = self.args.cats.lr
                    if 'xavier' not in mst:
                        mst['xavier'] = self.args.cats.xavier
                    if 'leaky' not in mst:
                        mst['leaky'] = self.args.cats.leaky
                    # if 'dropout' not in mst:
                    #     mst['dropout'] = self.args.cats.dropout
                    if 'batchnorm' not in mst:
                        mst['batchnorm'] = self.args.cats.batchnorm

                    if self.args.model == "gin":
                        mst['mlp_hidden'] = self.args.cats.mlp_hidden

                    optimizer_lookup[mst['optimizer']].append(i)
                    # mst['relu'] = True
                    kwargs = copy.deepcopy(mst)
                    kwargs['relu'] = False if layer == (
                        num_layers - 1) else True
                    kwargs['dropout'] = None if layer == (
                        num_layers - 1) else mst['dropout']
                    kwargs['batchnorm'] = False if layer == (
                        num_layers - 1) else mst['batchnorm']

                    del kwargs['optimizer']
                    del kwargs['lr']
                    args = [input_dim, output_dim]
                    all_args_list.append((args, kwargs))
                    logs("Init: {}, Index: {}".format(mst, i))
                print(all_args_list)
                layer_model = GCNBatchedLayer(
                    all_args_list, first_layer=(layer == 0))
                for i, submodel in enumerate(layer_model.models):

                    params_lookup[i] += list(submodel.parameters())
                    print("Submodel: {}, submodel param len: {}".format(
                        i, len(params_lookup[i])))
            else:
                if self.args.agg_pushdown:
                    #
                    if self.args.model == "gin":
                        from .nn import GINLayerAggPushDown as GCNLayer
                    else:
                        from .nn import GCNLayerAggPushDown as GCNLayer
                else:
                    print("Only GCN-non-pushdown is implemented")
                    from .nn import GCNLayer
                relu = False if layer == (num_layers - 1) else True
                dropout = None if layer == (
                    num_layers - 1) else self.args.cats.dropout
                batchnorm = False if layer == (
                    num_layers - 1) else self.args.cats.batchnorm

                if self.args.model == "gin":
                    layer_model = GCNLayer(
                        input_dim, output_dim, relu=relu,
                        dropout=dropout, mlp_hidden=self.args.cats.mlp_hidden)
                else:
                    layer_model = GCNLayer(
                        input_dim, output_dim, relu=relu,
                        dropout=dropout, batchnorm=batchnorm,
                        leaky=self.args.cats.leaky,
                        xavier=self.args.cats.xavier, device=self.device)
            # if not self.args.model_switch:
            layer_model = layer_model.to(self.device)
            self.models.append(layer_model)
        self.num_models = len(self.models)

        # layer_0 = GCNLayer(100, hidden)
        # layer_1 = GCNLayer(hidden, num_classes)
        # layer_0 = torch.nn.parallel.DistributedDataParallel(layer_0)
        # layer_1 = torch.nn.parallel.DistributedDataParallel(layer_1)

        if self.args.dataset == 'lognormal':
            self.criterion = torch.nn.CrossEntropyLoss()
        elif self.args.dataset == 'yelp':
            # multi-label
            self.criterion = torch.nn.BCEWithLogitsLoss(reduction='mean')
        else:
            self.criterion = torch.nn.CrossEntropyLoss(reduction='none')
        if self.args.dataset == 'yelp':
            self.logit_fn = torch.nn.Sigmoid()
            self.multi_label = True
        else:
            self.logit_fn = torch.nn.LogSoftmax()
            self.multi_label = False
        if self.args.lbfgs:
            self.params = []
            for layer in self.models:
                self.params += list(layer.parameters())
            self.optimizer = torch.optim.LBFGS(self.params, lr=1, max_iter=10)
        else:
            # if False:
            if self.args.lotan_model_batching:
                self.optimizers = []
                for i, mst in enumerate(self.args.cats.hs_msts):
                    if mst['optimizer'] == "adam":
                        optim_fn = torch.optim.AdamW
                    elif mst['optimizer'] == 'adagrad':
                        optim_fn = torch.optim.Adagrad
                    print(mst['lr'], optim_fn)
                    optimizer = optim_fn(
                        params_lookup[i], lr=mst['lr'],
                        weight_decay=self.args.cats.weight_decay,
                        eps=1e-12)
                    self.optimizers.append(optimizer)

            else:
                self.params = []
                for layer_model in self.models:
                    self.params += list(layer_model.parameters())
                # self.params = [{'params': model_layer.parameters()}
                #                for model_layer in self.models]

                print("Submodel: {}, submodel param len: {}".format(
                    0, len(self.params)))
                if self.args.cats.optimizer == "adam":
                    self.optimizer = torch.optim.AdamW(
                        self.params,
                        lr=self.args.cats.lr,
                        weight_decay=self.args.cats.weight_decay, eps=1e-12)
                elif self.args.cats.optimizer == "adagrad":
                    self.optimizer = torch.optim.Adagrad(
                        self.params,
                        lr=self.args.cats.lr,
                        weight_decay=self.args.cats.weight_decay,
                        lr_decay=0, eps=1e-12)
                self.optimizers = [self.optimizer]
                # self.optimizer = torch.optim.Adadelta(
                #   self.params, lr=0.03, weight_decay=self.args.cats.weight_decay)
                # self.optimizer = torch.optim.RMSprop(
                #   self.params,
                #   lr=0.03,
                #   weight_decay=self.args.cats.weight_decay,
                #   eps=1e-11, centered=True)
        if self.args.load_model != "":
            self.load_models()
        self.new_models = []

        for layer_model in self.models:
            layer_model = torch.nn.parallel.DistributedDataParallel(
                layer_model)
            self.new_models.append(layer_model)
        self.models = self.new_models

    def msg_emb(self, ident, batch_indices, result):
        objs = self.mini_batch_packed(batch_indices, result)
        return (ident, objs)

    def msg_grads(self, ident, batch_indices, V, U, H_u):
        objs = self.mini_batch_packed_grad(
            batch_indices, V, U, H_u.grad)
        return (ident, objs)

    def _mini_batch_forward(self, model, H_u, V, U, batch_indices, no_grad):
        H_u = torch.as_tensor(H_u, dtype=TORCH_DTYPE, device=self.device)
        if DEBUG:
            print("Torch Got: {}, Shape: {}".format(H_u, H_u.shape))
        H_u.requires_grad_()

        if no_grad is True:
            with torch.no_grad():
                result = model(
                    None, H_u, None, batch_indices, V, None, None)
        else:
            result = model(
                None, H_u, None, batch_indices, V, None, None)
        return H_u, result

    def mini_batch_forward(self, data, model, no_grad=False):
        ident, datum = data
        H_u, V, U, unique_vs, batch_indices = datum.H_u, datum.V, datum.U, datum.unique_vs, datum.batch_indices
        # V = torch.as_tensor(V, dtype=TORCH_DTYPE).to(self.device)
        # H_u = torch.tensor(H_u, dtype=TORCH_DTYPE, requires_grad=True)
        H_u, result = self._mini_batch_forward(
            model, H_u, V, U, batch_indices, no_grad)
        return (H_u, V, U), (ident, result), unique_vs, batch_indices

    def mini_batch_backward(self, data, model, layer_index=0):
        ident, datum = data
        grads, unique_vs, batch_indices = datum.grads, datum.unique_vs, datum.batch_indices
        (H_u, V, U) = self.layer_cache[layer_index][tuple(batch_indices)]
        H_u = H_u.to(self.device)
        H_u.requires_grad_()
        if layer_index != 0:
            H_u.requires_grad_()
        # V = V.to(self.device)
        H_v = model(
            None, H_u, None, unique_vs, V, None, None)
        grads = torch.as_tensor(grads).to(self.device)
        H_v.backward(grads)
        return (H_u, V, U), (ident, None), unique_vs, batch_indices

    def forward(self, *args, **kwargs):
        self.ward("forward", *args, **kwargs)

    def forbackward(self, *args, **kwargs):
        self.ward("forbackward", *args, **kwargs)

    def backward(self, *args, **kwargs):
        self.ward("backward", *args, **kwargs)

    def valid(self, model, data):
        ident_, datum_ = data
        _, valid_result = self._mini_batch_forward(
            model=model, H_u=datum_.H_u,
            V=datum_.V,
            U=datum_.U,
            batch_indices=datum_.batch_indices,
            no_grad=True)

        return valid_result

    def submodel_validation(self,
                            mode,
                            submodel_index,
                            valid_result, valid_labels):

        # print(valid_result)
        valid_result = self.logit_fn(valid_result)

        if self.multi_label:
            valid_result = valid_result.cpu().numpy()

            valid_labels = valid_labels.cpu().numpy()
            f1_micro, f1_macro = f1(valid_result, valid_labels)
            cm, bsize = cal_cm(valid_result, valid_labels)
            print("Submodel: {}, Mode: {}, F1s: {}".format(
                submodel_index, mode, (f1_micro, f1_macro)))
            correct = cm
        else:
            acc_packet = accuracy(
                valid_result,
                valid_labels,
                return_raw=True)[0]
            print("Submodel: {}, Mode: {}, Acc: {}".format(
                submodel_index, mode, acc_packet))
            correct, bsize = acc_packet
        if mode == 'valid':
            self.valid_correct[submodel_index] += correct
            self.valid_total[submodel_index] += bsize
            # acc = self.valid_correct / self.valid_total
        elif mode == 'test':
            self.test_correct[submodel_index] += correct
            self.test_total[submodel_index] += bsize
            # acc = self.test_correct / self.test_total

        # log_local_test_acc(self.args.rank,
        #                    self.epoch, mode, acc, self.total_eval)

    def validation(self, data, model, batch_indices, labels, mode='valid'):
        with logsc("VALIDATION", elapsed_time=True):
            if mode == 'valid':
                mask = self.meta[batch_indices, 4]
            elif mode == 'test':
                mask = self.meta[batch_indices, 5]
            valid_indices = []
            for idx, va in enumerate(mask):
                if va.item() == 1:
                    valid_indices.append(idx)

            if len(valid_indices) > 0:
                # print(valid_indices)
                model.eval()
                # H_u_valid = datum_.H_u[valid_indices, :]
                valid_result = self.valid(model, data)
                valid_result = valid_result[valid_indices, :]
                valid_labels = labels[valid_indices]
                # _, (_, valid_result), _ = \
                #     self.mini_batch_forward(
                #     (ident_, datum_), model, no_grad=True)
                if self.args.lotan_model_batching:
                    for i, (start, end) in enumerate(model.module.indices_output):
                        self.submodel_validation(
                            mode, i, valid_result[:, start:end], valid_labels)
                else:
                    self.submodel_validation(
                        mode, 0, valid_result, valid_labels)
                model.train()

    def pack_msg(
            self,
            direction,
            ident,
            batch_indices,
            V=None, U=None, H_u=None, result=None):
        with logsc(
            "{}_pack_msg_objs_building".format(direction), elapsed_time=True,
                log_dict=self.ward_log, accumulate=True):
            if direction in ["forbackward", "backward"]:
                if self.args.agg_pushdown:
                    # called msg_emb but actually passing grads
                    (ident, objs) = self.msg_emb(
                        ident, batch_indices, H_u.grad)
                else:
                    (ident, objs) = self.msg_grads(
                        ident, batch_indices, V, U, H_u)
            else:
                (ident, objs) = self.msg_emb(
                    ident, batch_indices, result)
        with logsc(
            "{}_pack_msg_serialize".format(direction), elapsed_time=True,
                log_dict=self.ward_log, accumulate=True):
            msg = self.serialize(objs)
        return (ident, msg)

    def data_preprocess(self, data):
        # only defined in shm implementation
        pass

    def data_postprocess(self, data):
        # only defined in shm implementation
        pass

    def cal_loss_and_backward(
            self,
            y,
            labels,
            train_mask,
            batch_indices, submodel_index=0):
        with logsc("FORWARD", elapsed_time=True):
            
            loss = self.criterion(y, labels)
            # logs(
            #   "Batch raw loss: {}, sum: {}".format(
            #   loss, loss.sum()))

            masked_loss = loss * train_mask
            # logs(
            #   "Masked loss: {}, sum: {}".format(
            #       masked_loss, masked_loss.sum()))

            # scaling to make sure it mathematically
            # equals single node execution
            # / train_mask.sum()
            #
            batch_loss = masked_loss.sum()
            fin_loss = batch_loss / self.total_train_count \
                * self.args.size
            if self.args.lbfgs:
                self.curr_epoch_total_loss += fin_loss
            logs("Submodel: {}, Batch loss: {}".format(
                submodel_index,
                fin_loss.detach().cpu().numpy()))
            logs("Submodel: {}, Batch raw loss: {}".format(
                submodel_index,
                batch_loss.detach().cpu().numpy()))
        if self.args.lotan_model_batching and \
                submodel_index != (len(self.args.cats.hs_msts) - 1):
            # last submodel cleans the graph
            retain_graph = True

        else:
            retain_graph = False
        # retain_graph = self.args.lotan_model_batching and (submodel_index !=
        logs("retain_graph: {}".format(retain_graph))
        with logsc("BACKWARD", elapsed_time=True):
            fin_loss.backward(retain_graph=retain_graph)

    def ward(self,
             direction,
             gen,
             model,
             outgoing_mq,
             first_layer=False,
             count_batches=False,
             layer_index=0):
        data = next(gen)
        self.direction = direction
        # the input must be transferred to the device where model resides
        if data is not None:

            ward_name = "WARD_{}".format(direction)
            with logsc(ward_name, elapsed_time=True,
                       log_dict=self.ward_log, accumulate=True):
                if DEBUG:
                    logs(data)
                self.data_preprocess(data)
                if direction == "forward":
                    with logsc(
                        "{}_mini_batch_forward".format(ward_name),
                        elapsed_time=True,
                            log_dict=self.ward_log, accumulate=True):
                        (H_u, V, U), (ident, result), \
                            unique_vs, batch_indices = self.mini_batch_forward(
                            data, model, no_grad=True)
                    if first_layer:
                        train_mask = self.meta[batch_indices, 3]
                        train_count = train_mask.sum()
                        self.partition_train_count += train_count
                    # always do a cache of the input, saved for backprop

                    self.layer_cache[layer_index][tuple(batch_indices)] = (
                        H_u.detach().to('cpu'), V, U)
                    # else:
                    #     self.layer_cache[layer_index][tuple(batch_indices)] = (
                    #         H_u.detach(), V, U)
                elif direction == "forbackward":
                    (H_u, V, U), (ident, result), unique_vs, batch_indices = self.mini_batch_forward(
                        data, model, no_grad=False)

                    batch_size = len(batch_indices)
                    if self.args.dataset == "lognormal":
                        # (dummy) loss compute
                        dummy_labels = torch.randint(
                            self.args.num_classes, (batch_size, ))
                        loss = self.criterion(result, dummy_labels)

                        loss.backward()
                    else:
                        with logsc("FORWARD", elapsed_time=True):
                            labels = self.meta[batch_indices, 1].to(
                                self.device)
                            train_mask = self.meta[batch_indices, 3].to(
                                self.device)
                            if self.args.lotan_model_batching:
                                for i, (start, end) in enumerate(
                                        model.module.indices_output):
                                    y = result[:, start:end]
                                    self.cal_loss_and_backward(
                                        y,
                                        labels,
                                        train_mask,
                                        batch_indices, submodel_index=i)
                            else:
                                self.cal_loss_and_backward(
                                    result, labels, train_mask, batch_indices)

                        # validation accuracy
                        with logsc("VALIDATION", elapsed_time=True):
                            self.validation(
                                data,
                                model,
                                batch_indices,
                                labels,
                                mode='valid')
                            self.validation(
                                data,
                                model,
                                batch_indices,
                                labels, mode='test')

                elif direction == "backward":
                    (H_u, V, U), (ident, result), unique_vs, batch_indices = self.mini_batch_backward(
                        data, model, layer_index)

                    self.ident_set.add(ident)

                # MSG Sending

                if direction == "backward" and first_layer:
                    if self.args.hard_partition:
                        msg = self.serialize(messages.ALL_FINISHED)
                        outgoing_mq.put((ident, msg))
                    else:
                        # backward first layer requires no msg sending
                        logs("No msg sending")
                else:

                    with logsc(
                        "{}_pack_msg".format(ward_name), elapsed_time=True,
                            log_dict=self.ward_log, accumulate=True):
                        (ident, msg) = self.pack_msg(
                            direction, ident,
                            unique_vs, V=V, U=U, H_u=H_u, result=result)
                    with logsc(
                        "{}_OMQ_enqueue".format(ward_name), elapsed_time=True,
                            log_dict=self.ward_log, accumulate=True):
                        logs("OMQ SIZE: {}".format(outgoing_mq.qsize()))
                        outgoing_mq.put((ident, msg))
                if count_batches:
                    self.total_batches += 1

                self.received += 1
                self.vertices_processed += len(batch_indices)
                logs(
                    "WORKER PROCESSED BATCHES: {self.received}/{self.total_batches}, VERTICES: {self.vertices_processed}".format(
                        **locals()))
                #
                self.data_postprocess(data)
                # del H_u, V, U, result
                # torch.cuda.empty_cache()
                # torch.cuda.synchronize()

    def save_models(self):
        other_states = {
            'epoch': self.epoch,
            'history': self.history
        }

        for i, layer_model in enumerate(self.models):
            name = os.path.join(self.args.save_model_root,
                                "layer_model_{}.pt".format(i))
            torch.save(layer_model.module.state_dict(), name)

        for i, optim in enumerate(self.optimizers):
            name = os.path.join(self.args.save_model_root,
                                "optimizer_{}.pt".format(i))
            torch.save(optim.state_dict(), name)

        name = os.path.join(self.args.save_model_root, "other_states.pt")
        torch.save(other_states, name)

    def load_models(self):
        for i, layer_model in enumerate(self.models):
            name = os.path.join(self.args.load_model,
                                "layer_model_{}.pt".format(i))
            layer_model.load_state_dict(torch.load(name))

        for i, optim in enumerate(self.optimizers):
            name = os.path.join(self.args.load_model,
                                "optimizer_{}.pt".format(i))
            optim.load_state_dict(torch.load(name))

        name = os.path.join(self.args.load_model, "other_states.pt")
        states = torch.load(name)
        self.epoch = states['epoch'] + 1
        self.history = states['history']

    def train_one_step_or_epoch(self, incoming_mq, outgoing_mq):
        self.curr_epoch_total_loss = 0
        self.valid_correct = defaultdict(lambda: 0)
        self.valid_total = defaultdict(lambda: 0)
        self.test_correct = defaultdict(lambda: 0)
        self.test_total = defaultdict(lambda: 0)
        self.valid_acc_all = defaultdict(lambda: 0)
        self.test_acc_all = defaultdict(lambda: 0)

        logs("GLOBAL_STEP: {}, EPOCH: {}".format(self.global_step, self.epoch))
        for optim in self.optimizers:
            optim.zero_grad()
        # self.optimizer.zero_grad()
        # layer cache
        self.layer_cache = defaultdict(dict)
        self.ward_log = {}
        # e.g. stages for models: [0, 1, 0]
        for j, model in enumerate(self.models + self.models[:-1][::-1]):
            self.ident_set = set()
            self.vertices_processed = 0
            self.received = 0

            if self.args.model_switch:
                model = model.to(self.device)

            if (self.epoch == 0) and (j == 0):
                # first epoch first layer counts number of vertices
                gen = self.nbgenerator(incoming_mq)
            else:
                gen = self.generator(incoming_mq)

            if (self.epoch == 0) and (j == 0):
                while self.stage.value == 0:
                    self.forward(
                        gen, model, outgoing_mq,
                        first_layer=True, count_batches=True, layer_index=j)

                # save the first layer keys
                # keys = list(self.first_layer_cache.keys())
                # with open('/local/keys.pickle', 'wb') as fp:
                #     pickle.dump(keys, fp)

                logs("First layer of first epoch finished")
            else:
                if j == self.num_models - 1:
                    direction = "forbackward"
                    func = self.forbackward
                    layer_idx = j
                elif j < self.num_models - 1:
                    direction = 'forward'
                    func = self.forward
                    layer_idx = j
                else:
                    direction = 'backward'
                    func = self.backward
                    layer_idx = (self.num_models + self.num_models - 2 - j)
                first_layer = (layer_idx == 0)
                with logsc(
                        "direction: {direction}, layer: {layer_idx}".format(
                            **locals())):
                    while self.received < self.total_batches:
                        if self.received == self.total_batches - 1:
                            # only syncs at the last minibatch
                            func(
                                gen,
                                model,
                                outgoing_mq,
                                first_layer=first_layer,
                                layer_index=layer_idx)
                        else:
                            logs("NO SYNC")
                            with model.no_sync():
                                func(
                                    gen,
                                    model,
                                    outgoing_mq,
                                    first_layer=first_layer,
                                    layer_index=layer_idx)

                logs(
                    "Finished, direction: {direction}, layer: {layer_idx}"
                    .format(**locals()))
                if DEBUG:
                    logs(self.ward_log)
            if self.args.model_switch:
                model = model.to("cpu")

        logs("One epoch finished")
        if self.args.rank == 0:
            logs("Saving models")
            self.save_models()

        for k, v in self.valid_correct.items():
            f1_arg = False
            if self.multi_label:
                valid_conf_mat = self.valid_correct[k]
                self.valid_correct[k] = valid_conf_mat.tolist()
                test_conf_mat = self.test_correct[k]
                self.test_correct[k] = test_conf_mat.tolist()

                valid_conf_mat_sum = valid_conf_mat.sum(axis=0)
                test_conf_mat_sum = test_conf_mat.sum(axis=0)

                valid_acc = f1_micro_from_cm(valid_conf_mat_sum)
                test_acc = f1_micro_from_cm(test_conf_mat_sum)
                f1_arg = True
                log_local_test_acc(
                    self.args.rank,
                    self.epoch,
                    "valid",
                    valid_acc,
                    self.valid_total[k], self.valid_correct[k], f1=f1_arg, submodel=k)
                log_local_test_acc(
                    self.args.rank,
                    self.epoch,
                    "test",
                    test_acc,
                    self.test_total[k], self.test_correct[k], f1=f1_arg, submodel=k)

            else:
                valid_acc = self.valid_correct[k] / self.valid_total[k] \
                    if self.valid_total[k] != 0 else 0
                test_acc = self.test_correct[k] / self.test_total[k] \
                    if self.test_total[k] != 0 else 0
                log_local_test_acc(
                    self.args.rank,
                    self.epoch,
                    "valid",
                    valid_acc,
                    self.valid_total[k], submodel=k)
                log_local_test_acc(
                    self.args.rank,
                    self.epoch,
                    "test",
                    test_acc,
                    self.test_total[k], submodel=k)

            self.valid_acc_all[k] = valid_acc
            self.test_acc_all[k] = test_acc
            
        self.return_message = "SUCCESS"

        msg = self.serialize(messages.ALL_FINISHED)
        # outgoing_mq.put((ident, msg))
        if not self.args.hard_partition:
            logs("Sending ALL_FINISHED to Graph engine")
            for i, ident in enumerate(list(self.ident_set)):
                if i == 0:
                    return_msg = self.serialize(self.return_message)
                    outgoing_mq.put((ident, return_msg))
                else:
                    outgoing_mq.put((ident, msg))

        self.history[self.epoch]['valid'] = dict(self.valid_acc_all)
        self.history[self.epoch]['test'] = dict(self.test_acc_all)
        self.history[self.epoch]['runtime'] = self.ward_log
        if DEBUG:
            logs(
                "Machine: {}, History: {}".format(
                    self.args.rank, dict(self.history)))
        self.epoch += 1
        del self.layer_cache
        gc.collect()

    def main(self, incoming_mq, outgoing_mq):
        self.device = self.get_device()
        self.dist_init()
        self.model_init()

        # hidden = 128
        # num_classes = 2

        # load meta
        if self.args.dataset != "lognormal":
            #
            with logsc("LOADING META", elapsed_time=True):
                self.meta = torch.tensor(np.loadtxt(os.path.join(
                    self.args.save_dir,
                    self.args.META),
                    delimiter=",",
                    dtype=int))

        self.total_batches = 0
        self.total_train_count = self.meta[:, 3].sum()
        logs("TOTAL_TRAIN_COUNT: {}".format(self.total_train_count))
        self.partition_train_count = 0
        self.history = defaultdict(dict)
        self.epoch = 0
        self.global_step = 0

        if self.args.lbfgs:
            for j in range(self.args.cats.lbfgs_steps):
                self.global_step = j

                def closure():
                    self.train_one_step_or_epoch(
                        incoming_mq, outgoing_mq)

                    return self.curr_epoch_total_loss

                self.optimizer.step(closure)
        else:

            for _ in range(self.args.cats.epochs):
                self.train_one_step_or_epoch(incoming_mq, outgoing_mq)
                for optim in self.optimizers:
                    optim.step()
                gc.collect()


class MPDummyWorker(IPCBase):
    """# multi-processed dummy Worker that takes in data to aggregate them
    """

    def subworker_main(self, incoming_mq, outgoing_mq):
        print("Subworker pid-{} started".format(os.getpid()))
        while True:
            # single-process feed to pytorch
            if self.verbose > 1:
                logs("Worker trying to dequeue imq ...")
            ident, msg = incoming_mq.get()
            if self.verbose > 1:
                logs("Dequeue succeeded ...")
            # with open('/tmp/msg_in_example.pickle', 'wb') as f:
            #     f.write(msg)
            obj = pickle.loads(msg)
            msg = pickle.dumps(plain_sum(obj))
            # with open('/tmp/msg_out_example.pickle', 'wb') as f:
            #     f.write(msg)
            # break
            if self.verbose > 1:
                logs("Worker trying to enqueue omg ...")
            outgoing_mq.put((ident, msg))
            if self.verbose > 1:
                logs("Enqueue succeeded ...")

    def worker_throughput_test_main(
            self, rounds, incoming_mq=None, outgoing_mq=None):
        logs("Subworker pid-{} started".format(os.getpid()))
        with open('msg_in_example.pickle', 'rb') as f:
            msg = f.read()
        for i in range(rounds):
            if self.args.router_imq_link_test:
                ident, msg = incoming_mq.get()
                # discard
            elif self.args.omq_router_link_test or self.args.omq_enque_test or self.args.omq_deque_test:
                payload = None if self.args.dummy_payload_test else msg
                outgoing_mq.put(('dummy_ident', payload))
            else:
                obj = pickle.loads(msg)
                msg = pickle.dumps(plain_sum(obj))
            if i % 5000 == 0:
                logs("subworker-{}, {}/{}, {}".format(
                    os.getpid(), i, rounds, len(msg)))

    def main(self, incoming_mq, outgoing_mq):
        processes = []

        with logsc("WORKERS PROCESSES", elapsed_time=True):
            for _ in range(self.args.num_workers):
                if self.args.worker_throughput_test\
                    or self.args.router_imq_link_test\
                        or self.args.omq_router_link_test\
                        or self.args.omq_enque_test \
                        or self.args.omq_deque_test:
                    total = 250000
                    proc = multiprocessing.Process(
                        target=self.worker_throughput_test_main, args=[
                            total // self.args.num_workers,
                            incoming_mq,
                            outgoing_mq
                        ])
                else:
                    proc = multiprocessing.Process(
                        target=self.subworker_main, args=[
                            incoming_mq, outgoing_mq])
                proc.start()
                processes.append(proc)
            for proc in processes:
                proc.join()


class PreBatchedWorkerSHM(PreBatchedWorker):
    def data_preprocess(self, data):
        data[1].unload()
        data[1].populate()

    def data_postprocess(self, data):
        data[1].clean()

    def mini_batch_packed(self, batch_indices, result):
        # `result` can be on GPU or CPU
        rlist = self.mini_batch_packed_cast(result)

        # drag batch_indices to 32bit
        batch_indices_32bit = batch_indices.astype(np.float32)

        local_arr = np.c_[batch_indices_32bit, rlist]
        if DEBUG:
            logs("Peek inside the return value: {}, shape: {}, dtype: {}".format(
                local_arr[0], local_arr.shape, local_arr.dtype))
        local_arr_ids = batch_indices
        if DEBUG:
            logs("Peek inside the return value ids: {}, shape: {}, dtype: {}".format(
                local_arr_ids, local_arr_ids.shape, local_arr_ids.dtype))

        shm_mem_name = "{}_{}".format(self.ident, self.received)
        shm_mem_name_ids = "{}_{}_ids".format(self.ident, self.received)

        shm_array, shm_mem = create_shm_array(shm_mem_name, local_arr)
        shm_array_ids, shm_mem_ids = create_shm_array(
            shm_mem_name_ids, local_arr_ids)

        ret = DatumSHM(
            shm_name=shm_mem_name,
            shm_name_ids=shm_mem_name_ids,
            shape=local_arr.shape,
            shape_ids=local_arr_ids.shape,
            H_u=1,
            V=None,
            U=None,
            unique_vs=0)
        shm_mem.close()

        return ret

    def mini_batch_packed_grad(self, batch_indices, V, U, H_u_grad):
        raise NotImplementedError
