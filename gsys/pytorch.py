import torch
import os
from .utils import logs
import datetime


# def init_ddp(args):
#     if args.gpu:
#         backend = 'nccl'
#         # backend = 'gloo'
#     else:
#         backend = 'gloo'

#     try:
#         rank = int(os.getenv('WORKER_NUMBER')) + 1
#     except Exception:
#         rank = 0

#     if args.dist:
#         torch.distributed.init_process_group(
#             backend=backend)
#     else:
#         torch.distributed.init_process_group(
#             backend=backend,
#             init_method='tcp://master:23456',
#             rank=rank, world_size=args.size)


def dist_init(args):
    try:
        rank = int(os.getenv('WORKER_NUMBER')) + 1
    except Exception:
        rank = 0
    args.rank = rank
    if args.gpu:
        backend = 'nccl'
    else:
        backend = 'gloo'
    logs(
        "DDP Initializing ... Rank: {}, World Size: {}".format(
            rank, args.size))
    if args.dist:
        # dist actually means "DGL"
        # 3 hrs timeout
        timeout = 3
        torch.distributed.init_process_group(
            backend=backend,
            timeout=datetime.timedelta(seconds=timeout * 3600))
    else:
        torch.distributed.init_process_group(
            backend=backend,
            init_method='tcp://master:23456',
            rank=rank, world_size=args.size)

    logs("DDP Initialized: {}".format(torch.distributed.is_initialized()))
