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

from gsys.data import Dataset
from gsys.constants import lg_products_dataset
from gsys.constants import constants
from gsys.all_args import get_all_args
import os
import torch
torch.manual_seed(constants.SEED)


# class args(object):
#     GNAME = 'ogbn-products'
#     MASKS = ['all_mask', 'train_mask', 'valid_mask', 'test_mask']
#     NTYPES = ['train', 'valid', 'test']
#     HOSTS = os.environ['ALL_HOSTS_DIR']
#     dataset = GNAME
#     self_loop = True
#     cats = lg_products_dataset
#     nine_one_random_split = True

if __name__ == "__main__":
    args = get_all_args()
    dataset = Dataset(args)
    dataset.read_dataset()
    dataset.dump_spark()
