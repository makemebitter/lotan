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

import argparse
from types import SimpleNamespace
from .constants import sage_cats
from .constants import gcn_cats
from .constants import gin_cats
from .constants import products_dataset
from .constants import papers100M_dataset
from .constants import arxiv_dataset
from .constants import lognormal_dataset
from .constants import yelp_dataset
from .constants import reddit_dataset
from .constants import amazon_dataset
from .utils import logs
import os


def get_all_args(no_stdout=False):
    parser = get_main_parser()
    pargs = parser.parse_args()
    args = get_args(pargs, no_stdout)
    if not no_stdout:
        logs("ALL ARGS: {}".format(args))
        cats = SimpleNamespace(**args.cats._dict())
        logs("ALL CATS: {}".format(cats))
    return args


def get_args(pargs, no_stdout=False):
    if pargs.dataset == 'lognormal':
        args = lognormal_dataset()
    elif pargs.dataset == 'ogbn-products':
        args = products_dataset()
    elif pargs.dataset == 'ogbn-papers100M':
        args = papers100M_dataset()
    elif pargs.dataset == 'yelp':
        args = yelp_dataset()
    elif pargs.dataset == 'reddit':
        args = reddit_dataset()
    elif pargs.dataset == 'ogbn-arxiv':
        args = arxiv_dataset()
    elif pargs.dataset == 'amazon':
        args = amazon_dataset()
    args = SimpleNamespace(**args._dict(), **pargs.__dict__)
    try:
        rank = int(os.getenv('WORKER_NUMBER')) + 1
    except Exception:
        rank = 0
    args.rank = rank

    if args.model == 'sage':
        args.cats = sage_cats()
    elif args.model == 'gcn':
        args.cats = gcn_cats()
    elif args.model == 'gin':
        args.cats = gin_cats()

    overwrite_cats(pargs, args, no_stdout=no_stdout)

    return args


def overwrite_cats(pargs, args, no_stdout=False):
    if pargs.model_lr is not None:
        args.cats.lr = pargs.model_lr
    if pargs.model_num_hidden is not None:
        args.cats.num_hidden = pargs.model_num_hidden
    if pargs.model_num_layers is not None:
        args.cats.num_layers = pargs.model_num_layers
    if pargs.model_batch_size is not None:
        args.cats.batch_size = pargs.model_batch_size
    if pargs.model_weight_decay is not None:
        args.cats.weight_decay = pargs.model_weight_decay
    if pargs.model_dropout is not None:
        args.cats.dropout = pargs.model_dropout
    if pargs.model_epochs is not None:
        args.cats.epochs = pargs.model_epochs
    if pargs.model_optimizer is not None:
        args.cats.optimizer = pargs.model_optimizer
    if pargs.model_batchnorm is not None:
        args.cats.batchnorm = pargs.model_batchnorm == "True"
    if pargs.model_leaky is not None:
        args.cats.leaky = pargs.model_leaky == "True"
    if pargs.model_xavier is not None:
        args.cats.xavier = pargs.model_xavier == "True"
    if pargs.model_mlp_hidden is not None:
        args.cats.mlp_hidden = pargs.model_mlp_hidden
    args.cats.refresh()

    if pargs.drill_down_mb_size:
        # lr_list = [0.01,
        #            0.05,
        #            0.001,
        #            0.005,
        #            0.0001,
        #            0.0005,
        #            0.00001,
        #            0.00005,
        #            0.00002,
        #            0.00003,
        #            0.00004,
        #            0.00006,
        #            0.00007,
        #            0.00008,
        #            0.00009,
        #            0.0002,
        #            0.0003,
        #            0.0004,
        #            0.0005,
        #            0.0006
        #            ]
        # hs_grid = {
        #     'optimizer': ["adam"],
        #     'lr': lr_list[:pargs.drill_down_mb_size],
        #     'dropout': [0]
        # }
        # args.cats.hs_grid = hs_grid
        base_len = len(args.cats.hs_msts)
        wrap = pargs.drill_down_mb_size // base_len
        around = pargs.drill_down_mb_size % base_len
        start = pargs.drill_down_mb_below_batching_start
        args.cats.hs_msts = args.cats.hs_msts * wrap \
            + args.cats.hs_msts[start:start+around]
    if not no_stdout:
        logs(args.cats.hs_msts)


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def get_main_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--partition', type=int, default=1
    )
    parser.add_argument(
        '--prepare_dataset', action='store_true'
    )
    parser.add_argument(
        '--train', action='store_true'
    )
    parser.add_argument(
        '--gpu', action='store_true'
    )
    parser.add_argument(
        '--self_loop', action='store_true'
    )
    parser.add_argument(
        '--undirected', action='store_true'
    )
    parser.add_argument(
        '--dist', action='store_true'
    )
    parser.add_argument(
        '--size', type=int, default=2
    )
    parser.add_argument(
        '--verbose', type=int, default=2
    )
    parser.add_argument(
        '--local_rank', type=int, default=0
    )
    parser.add_argument(
        '--model', type=str, default='sage'
    )
    parser.add_argument(
        '--master', type=str, default='master'
    )
    parser.add_argument(
        '--dataset', type=str, default='ogbn-products'
    )
    parser.add_argument(
        '--worker_type', type=str, default='worker'
    )
    parser.add_argument(
        '--num_receiver_senders', type=int, default=1
    )
    parser.add_argument(
        '--mini_batch_size', type=int, default=100
    )
    parser.add_argument(
        '--num_workers', type=int, default=10
    )
    parser.add_argument(
        '--cpu_cores_per_machine', type=int, default=40
    )
    parser.add_argument(
        '--network_arch', type=str, default='async'
    )
    parser.add_argument(
        '--io_type', type=str, default='json'
    )
    parser.add_argument(
        '--ipc_type', type=str, default='socket'
    )
    parser.add_argument(
        '--send_finished', action='store_true'
    )
    parser.add_argument(
        '--send_term', action='store_true'
    )
    parser.add_argument(
        '--worker_throughput_test', action='store_true'
    )
    parser.add_argument(
        '--router_throughput_test', action='store_true'
    )
    parser.add_argument(
        '--messenger_router_link_test', action='store_true'
    )
    parser.add_argument(
        '--router_imq_link_test', action='store_true'
    )
    parser.add_argument(
        '--omq_router_link_test', action='store_true'
    )
    parser.add_argument(
        '--router_messenger_link_test', action='store_true'
    )
    parser.add_argument(
        '--omq_enque_test', action='store_true'
    )
    parser.add_argument(
        '--omq_deque_test', action='store_true'
    )
    parser.add_argument(
        '--dummy_payload_test', action='store_true'
    )
    parser.add_argument(
        '--messenger_final_backprop', action='store_true'
    )
    parser.add_argument(
        '--messenger_single_vector', action='store_true'
    )
    parser.add_argument(
        '--messenger_plain_forbackward', action='store_true'
    )
    parser.add_argument(
        '--messenger_backprop', action='store_true'
    )
    parser.add_argument(
        '--messenger_idx', type=str, default=None
    )
    parser.add_argument(
        '--agg_pushdown', type=str2bool, default=True
    )
    parser.add_argument(
        '--plain_direction', type=str, default=None
    )
    parser.add_argument(
        '--nine_one_random_split', action='store_true'
    )
    parser.add_argument(
        '--k_hop', type=int, default=1
    )
    parser.add_argument(
        '--ali_full_batch', action='store_true'
    )
    parser.add_argument(
        '--sparse', action='store_true'
    )
    parser.add_argument(
        '--first_sparse_pipe', action='store_true'
    )
    parser.add_argument(
        '--model_lr', type=float, default=None
    )
    parser.add_argument(
        '--model_num_hidden', type=int, default=None
    )
    parser.add_argument(
        '--model_num_layers', type=int, default=None
    )
    parser.add_argument(
        '--model_batch_size', type=int, default=None
    )
    parser.add_argument(
        '--model_weight_decay', type=float, default=None
    )
    parser.add_argument(
        '--model_dropout', type=float, default=None
    )
    parser.add_argument(
        '--model_epochs', type=int, default=None
    )
    parser.add_argument(
        '--model_optimizer', type=str, default=None
    )
    parser.add_argument(
        '--model_batchnorm', type=str, default=None
    )
    parser.add_argument(
        '--model_leaky', type=str, default=None
    )
    parser.add_argument(
        '--model_xavier', type=str, default=None
    )
    parser.add_argument(
        '--model_mlp_hidden', type=int, nargs='+', default=None
    )
    parser.add_argument(
        '--model_switch', action='store_true'
    )
    parser.add_argument(
        '--lbfgs', action='store_true'
    )
    parser.add_argument(
        '--lotan_model_batching', action='store_true'
    )
    parser.add_argument(
        '--spark_worker_cores', type=int, default=40
    )
    parser.add_argument(
        '--hard_partition', action='store_true'
    )
    parser.add_argument(
        '--drill_down_mb_size', type=int, default=None
    )
    parser.add_argument(
        '--drill_down_mb_below_batching_start', type=int, default=0
    )
    parser.add_argument(
        '--save_model_root', type=str, default="/mnt/nfs/models"
    )
    parser.add_argument(
        '--load_model', type=str, default=""
    )

    return parser
