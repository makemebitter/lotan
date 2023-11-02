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
    # HOSTS = '/mnt/nfs/all_host_list'
    FEATURE = 'feat'
    LABELS = 'labels'
    PIPES_ROOT = '/tmp'
    M2W = 'm2w'
    W2M = 'w2m'
    SEED = 2021
    DGL_DATASET_DIR = '/mnt/nfs/datasets'
    NFS_ROOT = '/mnt/nfs/ssd'
    DGL_CACHE_DIR = os.path.join(NFS_ROOT, 'dgl_cached')
    # MACHINES = [
    #     "10.0.1.{}:5699".format(i) for i in range(1, 5)
    # ]


class network_constants(constants):
    ROUTER_ADDRESS = f'ipc://{constants.PIPES_ROOT}/router'
    CONTROLLER_ADDRESS = f'ipc://{constants.PIPES_ROOT}/controller'
    ROUTER_BACKEND_IPC = f'ipc://{constants.PIPES_ROOT}/backend'
    ROUTER_BACKEND_IMQ_IPC = f'ipc://{constants.PIPES_ROOT}/backend_imq'
    ROUTER_BACKEND_OMQ_IPC = f'ipc://{constants.PIPES_ROOT}/backend_omq'
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


def dgl_class_factory(GNAME, name):
    created_class = type(name, (object, ), {
        "GNAME": GNAME,
        "DATA_REPO": f'{constants.DGL_DATASET_DIR}/{GNAME}',
        "DATA_REPO_NFS": f'{constants.DGL_DATASET_DIR}/{GNAME}',
        "cache_dir": f'{constants.DGL_CACHE_DIR}/{GNAME}'
    }
    )
    return created_class


dgl_products_dataset = dgl_class_factory(
    "ogbn-products", "dgl_products_dataset")
dgl_arxiv_dataset = dgl_class_factory("ogbn-arxiv", "dgl_arxiv_dataset")
dgl_papers100M_dataset = dgl_class_factory(
    "ogbn-papers100M", "dgl_papers100M_dataset")
dgl_yelp_dataset = dgl_class_factory(
    "yelp", "dgl_yelp_dataset")
dgl_reddit_dataset = dgl_class_factory(
    "reddit", "dgl_reddit_dataset")
dgl_amazon_dataset = dgl_class_factory(
    "amazon", "dgl_amazon_dataset")


class dgl_lognormal_dataset:
    GNAME = 'lognormal'
    EDGE_FILE_PATH = f"{constants.NFS_ROOT}/lognormal/lognormal_edge.txt"
    VERTEX_FILE_PATH = f"{constants.NFS_ROOT}/lognormal/lognormal_vertex.txt"
    DATA_REPO = f'{constants.DGL_DATASET_DIR}/lognormal'
    DATA_REPO_NFS = f'{constants.DGL_DATASET_DIR}/lognormal'


class ali_products_dataset:
    ALI_DATA_REPO = f'{constants.NFS_ROOT}/ogbn-products-ali'


class ali_arxiv_dataset:
    ALI_DATA_REPO = f'{constants.NFS_ROOT}/ogbn-arxiv-ali'

class ali_yelp_dataset:
    ALI_DATA_REPO = f'{constants.NFS_ROOT}/yelp-ali'

class ali_reddit_dataset:
    ALI_DATA_REPO = f'{constants.NFS_ROOT}/reddit-ali'

class ali_amazon_dataset:
    ALI_DATA_REPO = f'{constants.NFS_ROOT}/amazon-ali'

class ali_papers100M_dataset:
    ALI_DATA_REPO = f'{constants.NFS_ROOT}/ogbn-papers100M-ali'


class lg_dataset_base(Constants):
    FEATURES = "features.csv"
    EDGES = "edges.csv"
    META = "meta.csv"


class lg_products_dataset(lg_dataset_base):
    save_dir = f"{constants.NFS_ROOT}/products"
    feature_shape = 100
    num_classes = 47


class lg_arxiv_dataset(lg_dataset_base):
    save_dir = f"{constants.NFS_ROOT}/arxiv"
    feature_shape = 128
    num_classes = 40

class lg_yelp_dataset(lg_dataset_base):
    save_dir = f"{constants.NFS_ROOT}/yelp"
    feature_shape = 300
    num_classes = 100

class lg_reddit_dataset(lg_dataset_base):
    save_dir = f"{constants.NFS_ROOT}/reddit"
    feature_shape = 602
    num_classes = 41

class lg_amazon_dataset(lg_dataset_base):
    save_dir = f"{constants.NFS_ROOT}/amazon"
    feature_shape = 300
    num_classes = 22


class lg_papers100M_dataset(lg_dataset_base):
    save_dir = f"{constants.NFS_ROOT}/papers100M"
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
