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



# for DGL usage only
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
        timeout = 30000
        torch.distributed.init_process_group(
            backend=backend,
            timeout=datetime.timedelta(seconds=timeout * 3600))
    else:
        torch.distributed.init_process_group(
            backend=backend,
            init_method='tcp://master:23456',
            rank=rank, world_size=args.size)

    logs("DDP Initialized: {}".format(torch.distributed.is_initialized()))
