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

import itertools
import os
LARGE_INT = 1e10
TIME_FORMAT = '%Y-%m-%d %H:%M:%S'


class Constants(object):
    def __str__(self):
        dic = self._dict()

        return str(dic)

    def _dict(self):
        return {
            att: getattr(
                self, att) for att in dir(self) if not att.startswith('_')}

    # def __repr__(self):
    #     return self.__str__()


class constants(Constants):
    MASKS = ['all_mask', 'train_mask', 'valid_mask', 'test_mask']
    NTYPES = ['train', 'valid', 'test']
    HOSTS = '/mnt/nfs/all_host_list'
    FEATURE = 'feat'
    LABELS = 'labels'
    PIPES_ROOT = '/mnt/ssd/tmp'
    M2W = 'm2w'
    W2M = 'w2m'
    SEED = 2021
    MACHINES = [
        "10.0.1.{}:5699".format(i) for i in range(1, 5)
    ]


class network_constants(constants):
    ROUTER_ADDRESS = 'ipc:///mnt/ssd/tmp/router'
    CONTROLLER_ADDRESS = 'ipc:///mnt/ssd/tmp/controller'
    ROUTER_BACKEND_IPC = 'ipc:///mnt/ssd/tmp/backend'
    ROUTER_BACKEND_IMQ_IPC = 'ipc:///mnt/ssd/tmp/backend_imq'
    ROUTER_BACKEND_OMQ_IPC = 'ipc:///mnt/ssd/tmp/backend_omq'
    INCOMING_MQ_MAX_SIZE = 20
    OUTCOMING_MQ_MAX_SIZE = INCOMING_MQ_MAX_SIZE
    INCOMING_MQ_TIMEOUT = 2
    ZMQ_RCVHWM = INCOMING_MQ_MAX_SIZE
    ZMQ_SNDHWM = INCOMING_MQ_MAX_SIZE
    ZMQ_IO_THREADS = 1


# cats base


class cats_base(constants):
    num_hidden = 256
    num_layers = 2
    dropout = 0.5
    lr = 0.05
    epochs = 300
    lbfgs_steps = 100
    lbfgs_max_iters = 10
    sampling = [25, 10]
    optimizer = "adam"
    # , 'adagrad'
    # run: adam all
    hs_grid = {
        'optimizer': ["adam", "adagrad"],
        'lr': [0.01, 0.05],
        'dropout': [0, 0.5]
    }
    # 0.05 , "adagrad"

    def refresh(self):
        self.hs_msts = [{
            "optimizer": x[0],
            "lr": x[1],
            "dropout": x[2]
        } for x in list(
            itertools.product(self.hs_grid['optimizer'], self.hs_grid['lr'],
                              self.hs_grid['dropout']))]


# minibatch


class sage_cats(cats_base):
    pass


class gcn_cats(cats_base):
    num_hidden = 256
    num_layers = 3
    batch_size = 2048
    weight_decay = 0
    dropout = 0.5
    # batch_size = LARGE_INT
    mini_batch_size = 1000
    mlp_hidden = [128]
    batchnorm = False
    leaky = True
    xavier = False

    def __init__(self):
        self.refresh()

    def refresh(self):
        super(gcn_cats, self).refresh()
        self.sampling = [None for _ in range(self.num_layers)]

# class gcn_cats(cats_base):
#     num_hidden = 128
#     num_layers = 2
#     batch_size = 2048
#     weight_decay = 0
#     # full batch
#     sampling = [None, None]
#     # batch_size = LARGE_INT
#     mini_batch_size = 1000


# class gin_cats(cats_base):
#     pass


class gin_cats(gcn_cats):
    num_hidden = 256
    mlp_hidden = [128]


class dgl_products_dataset:
    GNAME = 'ogbn-products'
    DATA_REPO = '/mnt/nfs/datasets/ogbn-products'
    DATA_REPO_NFS = '/mnt/nfs/datasets/ogbn-products'
    cache_dir = '/mnt/nfs/ssd/dgl_cached/ogbn-products'


class dgl_arxiv_dataset:
    GNAME = 'ogbn-arxiv'
    DATA_REPO = '/mnt/nfs/datasets/ogbn-arxiv'
    DATA_REPO_NFS = '/mnt/nfs/datasets/ogbn-arxiv'
    cache_dir = '/mnt/nfs/ssd/dgl_cached/ogbn-arxiv'


class dgl_papers100M_dataset:
    GNAME = 'ogbn-papers100M'
    DATA_REPO = '/mnt/nfs/datasets/ogbn-papers100M'
    DATA_REPO_NFS = '/mnt/nfs/datasets/ogbn-papers100M'
    cache_dir = '/mnt/nfs/dgl_cached/ogbn-papers100M'

class dgl_yelp_dataset:
    GNAME = 'yelp'
    DATA_REPO = '/mnt/nfs/datasets/yelp'
    DATA_REPO_NFS = '/mnt/nfs/datasets/yelp'
    cache_dir = '/mnt/nfs/ssd/dgl_cached/yelp'

class dgl_reddit_dataset:
    GNAME = 'reddit'
    DATA_REPO = '/mnt/nfs/datasets/reddit'
    DATA_REPO_NFS = '/mnt/nfs/datasets/reddit'
    cache_dir = '/mnt/nfs/ssd/dgl_cached/reddit'

class dgl_amazon_dataset:
    GNAME = 'amazon'
    DATA_REPO = '/mnt/nfs/datasets/amazon'
    DATA_REPO_NFS = '/mnt/nfs/datasets/amazon'
    cache_dir = '/mnt/nfs/ssd/dgl_cached/amazon'

class dgl_lognormal_dataset:
    GNAME = 'lognormal'
    EDGE_FILE_PATH = "/mnt/nfs/ssd/lognormal/lognormal_edge.txt"
    VERTEX_FILE_PATH = "/mnt/nfs/ssd/lognormal/lognormal_vertex.txt"
    DATA_REPO = '/mnt/nfs/datasets/lognormal'
    DATA_REPO_NFS = '/mnt/nfs/datasets/lognormal'


class ali_products_dataset:
    ALI_DATA_REPO = '/mnt/nfs/ogbn-products-ali'


class ali_arxiv_dataset:
    ALI_DATA_REPO = '/mnt/nfs/ogbn-arxiv-ali'

class ali_yelp_dataset:
    ALI_DATA_REPO = '/mnt/nfs/yelp-ali'

class ali_reddit_dataset:
    ALI_DATA_REPO = '/mnt/nfs/reddit-ali'

class ali_amazon_dataset:
    ALI_DATA_REPO = '/mnt/nfs/amazon-ali'

class ali_papers100M_dataset:
    ALI_DATA_REPO = '/mnt/nfs/ogbn-papers100M-ali'


class lg_dataset_base(Constants):
    FEATURES = "features.csv"
    EDGES = "edges.csv"
    META = "meta.csv"


class lg_products_dataset(lg_dataset_base):
    save_dir = "/mnt/nfs/ssd/products"
    feature_shape = 100
    num_classes = 47


class lg_arxiv_dataset(lg_dataset_base):
    save_dir = "/mnt/nfs/ssd/arxiv"
    feature_shape = 128
    num_classes = 40

class lg_yelp_dataset(lg_dataset_base):
    save_dir = "/mnt/nfs/ssd/yelp"
    feature_shape = 300
    num_classes = 100

class lg_reddit_dataset(lg_dataset_base):
    save_dir = "/mnt/nfs/ssd/reddit"
    feature_shape = 602
    num_classes = 41

class lg_amazon_dataset(lg_dataset_base):
    save_dir = "/mnt/nfs/ssd/amazon"
    feature_shape = 300
    num_classes = 22


class lg_papers100M_dataset(lg_dataset_base):
    save_dir = "/mnt/nfs/papers100M"
    feature_shape = 128
    num_classes = 172


class lg_lognormal_dataset(lg_dataset_base):
    feature_shape = 100
    num_classes = 2


class products_dataset(
        lg_products_dataset,
        dgl_products_dataset,
        ali_products_dataset):
    pass


class arxiv_dataset(
    lg_arxiv_dataset,
    dgl_arxiv_dataset,
    ali_arxiv_dataset
):
    pass


class papers100M_dataset(
    lg_papers100M_dataset,
    dgl_papers100M_dataset,
    ali_papers100M_dataset
):
    pass

class yelp_dataset(
    lg_yelp_dataset,
    dgl_yelp_dataset,
    ali_yelp_dataset
):
    pass

class reddit_dataset(
    lg_reddit_dataset,
    dgl_reddit_dataset,
    ali_reddit_dataset
):
    pass

class amazon_dataset(
    lg_amazon_dataset,
    dgl_amazon_dataset,
    ali_amazon_dataset
):
    pass


class lognormal_dataset(lg_lognormal_dataset, dgl_lognormal_dataset):
    pass
