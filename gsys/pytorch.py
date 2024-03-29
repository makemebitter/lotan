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
from .all_args import get_rank

# for DGL usage only
def dist_init(args):
    rank = get_rank()
    args.rank = rank
    if torch.cuda.is_available() and self.args.gpu:
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
