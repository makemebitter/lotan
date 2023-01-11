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
