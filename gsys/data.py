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

from .utils import timeit_factory
from .utils import logs
from ogb.nodeproppred import DglNodePropPredDataset
import glob
import csv
import torch
from gsys.constants import constants
# from gsys.ipc import CHAR_MOD
import os
import numpy as np
import dgl
from pathlib import Path
from tqdm import tqdm


def gen_file(files):
    for file in files:
        print(file)
        with open(file) as f:
            for line in f:
                yield line


class Dataset(object):
    def __init__(self, args):
        self.args = args

    @timeit_factory()
    def read_dataset(self, force=False):
        logs("Reading dataset".format())
        self.read_node_dataset(force)

    def update_masks(self):
        all_mask = torch.zeros(self.g.num_nodes(), dtype=torch.long)
        train_mask = torch.zeros(self.g.num_nodes(), dtype=torch.long)
        valid_mask = torch.zeros(self.g.num_nodes(), dtype=torch.long)
        test_mask = torch.zeros(self.g.num_nodes(), dtype=torch.long)
        for i, ntype in enumerate(constants.NTYPES):
            idx = self.split_idx[ntype]
            if idx.shape[0] != 0:
                all_mask[idx] = i
                for mask, ori_ntype in zip(
                        [train_mask, valid_mask, test_mask], constants.NTYPES):
                    if ntype == ori_ntype:
                        mask[idx] = 1

        for mask_name, mask in zip(
                constants.MASKS, [
                    all_mask, train_mask, valid_mask, test_mask]):
            self.g.ndata[mask_name] = mask

    def read_spark_graph(self):
        all_edges = sorted(glob.glob(
            os.path.join(self.args.EDGE_FILE_PATH, 'part*')))
        all_vertices = sorted(glob.glob(
            os.path.join(self.args.VERTEX_FILE_PATH, 'part*')))
        edge_arr = np.asarray(list(csv.reader(gen_file(all_edges))), dtype=int)
        vertex_arr = np.asarray(
            list(
                csv.reader(gen_file(all_vertices))), dtype=int)
        vertex_arr = vertex_arr.reshape(
            vertex_arr.size)
        self.g = dgl.graph((edge_arr[:, 0], edge_arr[:, 1]))
        self.g.ndata[constants.FEATURE] = torch.randn(self.g.num_nodes(), 100)
        num_classes = 47
        self.g.ndata[constants.LABELS] = torch.randint(
            0, num_classes, (self.g.num_nodes(), ))
        split_idx = {
            "train": torch.arange(0, self.g.num_nodes() - 20),
            'valid': torch.arange(
                self.g.num_nodes() - 20, self.g.num_nodes() - 10),
            'test': torch.arange(self.g.num_nodes() - 10, self.g.num_nodes())
        }
        self.split_idx = split_idx
        return self.g, self.split_idx

    def random_split(self, p):
        nodes = self.g.nodes()
        torch.manual_seed(constants.SEED)
        rand_mask = torch.rand(nodes.shape)
        train_idx = (rand_mask < p).nonzero(as_tuple=True)[0]

        test_idx = (rand_mask > p).nonzero(as_tuple=True)[0]
        self.split_idx = {
            'train': train_idx,
            'valid': torch.tensor([]),
            'test': test_idx
        }

    def get_cache_path(self):
        cache_path = os.path.join(self.args.cache_dir, "graph.bin")
        return cache_path

    def read_node_dataset(self, force=False):
        cache_path = self.get_cache_path()
        if os.path.exists(cache_path) and not force:
            logs("DGL cached loading")
            self.g = dgl.data.utils.load_graphs(cache_path, [0])[0][0]
        else:

            logs("DGL fresh reading")
            if self.args.dataset == 'lognormal':
                self.read_spark_graph()
            if self.args.dataset in ['yelp', 'reddit']:
                if self.args.dataset == 'yelp':
                    DGLDatasetOBJ = dgl.data.YelpDataset
                else:
                    DGLDatasetOBJ = dgl.data.RedditDataset
                self.dataset = DGLDatasetOBJ(raw_dir=self.args.cache_dir)
                self.g = self.dataset[0]
                self.split_idx = {}
                for mask_name, split_name in zip(["train_mask", "val_mask", "test_mask"], constants.NTYPES):
                    split = self.g.ndata[mask_name].nonzero(as_tuple=True)[0]
                    self.split_idx[split_name] = split
                if self.args.dataset == 'yelp':
                    self.g.ndata['labels'] = self.g.ndata['label'].float()
                else:
                    self.g.ndata['labels'] = self.g.ndata['label']
            else:
                # OGB datasets
                logs("DGL Reading")
                self.dataset = DglNodePropPredDataset(name=self.args.GNAME)
                logs("DGL get_idx_split")
                self.split_idx = self.dataset.get_idx_split()
                self.g, self.labels = self.dataset[0]
                self.g.ndata['labels'] = torch.nan_to_num(
                    self.labels[:, 0], 0).long()
            if self.args.nine_one_random_split:
                logs("Random split")
                self.random_split(p=0.9)
            logs("Update masks")
            self.update_masks()
            self.cache_save()

        if self.args.dataset == 'amazon':
            self.split_idx = {}
            for mask_name, split_name in zip(["train_mask", "valid_mask", "test_mask"], constants.NTYPES):
                split = self.g.ndata[mask_name].nonzero(as_tuple=True)[0]
                self.split_idx[split_name] = split

        if self.args.undirected:
            logs("Make bidirected")
            self.g = dgl.to_bidirected(self.g, copy_ndata=True)
        if self.args.self_loop:
            logs("Add self loop")
            self.g = dgl.add_self_loop(self.g)
            

        self.feature_shape = self.g.ndata[constants.FEATURE].shape[1]

        logs("DGL get num classes")
        if self.args.dataset == "ogbn-papers100M":
            self.num_classes = self.args.num_classes
        elif self.args.dataset in ["yelp"]:
            self.num_classes = self.g.ndata[constants.LABELS].shape[1]
        else:
            self.num_classes = len(
                torch.unique(
                    self.g.ndata[constants.LABELS][
                        0:self.g.number_of_nodes()]))

    def cache_save(self):
        cache_path = self.get_cache_path()
        logs("Cache DGL")
        Path(self.args.cache_dir).mkdir(parents=True, exist_ok=True)
        dgl.data.utils.save_graphs(cache_path, [self.g])

    @timeit_factory()
    def dump(self):
        dgl.distributed.partition_graph(
            self.g,
            self.args.GNAME,
            self.args.partition,
            self.args.DATA_REPO_NFS,
            balance_ntypes=self.g.ndata[constants.MASKS[0]])

    @timeit_factory()
    def khop_filter(self, k, g):
        nodes = g.nodes()
        mask_seeds = g.ndata['train_mask'] + \
            g.ndata['valid_mask'] + g.ndata['test_mask']
        mask_seeds = mask_seeds.astype('bool')
        seeds = nodes[mask_seeds]
        return dgl.khop_in_subgraph(g, seeds, k)

    @timeit_factory()
    def khop_filter_dump(self, k):
        self.args.save_dir = self.ori_save_dir + "_{}hop".format(k)
        self.g = self.khop_filter(k, self.ori_g)
        self._dump_spark()

    @timeit_factory()
    def dump_spark(self):
        if self.args.dataset == "ogbn-papers100M":
            self.ori_save_dir = self.args.save_dir
            self.ori_g = self.g
            # self.khop_filter_dump(2)
            self.khop_filter_dump(3)
            # self.khop_filter_dump(4)
        else:
            self._dump_spark()

    def _dump_spark(self):
        save_dir = self.args.save_dir
        Path(save_dir).mkdir(parents=True, exist_ok=True)

        labels = self.g.ndata['labels']
        labels = torch.nan_to_num(labels, 0)

    #     labels = self.g.ndata['labels']
    #     nodes = g.nodes().numpy().astype(int)
        nodes = self.g.nodes()
        meta = torch.column_stack(
            (
                nodes,
                labels,
                self.g.ndata['all_mask'],
                self.g.ndata['train_mask'],
                self.g.ndata['valid_mask'],
                self.g.ndata['test_mask'])
        )
        labeled_mask = self.g.ndata['train_mask'] + \
            self.g.ndata['valid_mask'] + self.g.ndata['test_mask']
        logs("Saving meta".format())
        np.savetxt(os.path.join(save_dir, "meta.csv"),
                   meta, delimiter=",", fmt='%i')

        del meta

        feat = self.g.ndata['feat']

        logs("Saving features")
        feature_path = os.path.join(save_dir, "features.csv")
        if self.args.dataset == "ogbn-papers100M":
            fmt = '%.4e'

        else:
            fmt = '%.18e'

        with open(feature_path, 'w') as the_file:
            for i in tqdm(range(nodes.shape[0])):
                node = nodes[i]
                fea = feat[i]
                labeled = labeled_mask[i]
                to_write = str(node.item()) + "," + str(labeled.item()
                                                        ) + "," + ','.join(np.char.mod("%.4e", fea))
                the_file.write(to_write + '\n')

    #     np.savetxt(
    #       os.path.join(
    #           save_dir,
    #           "labels.csv"), labels.numpy(), delimiter=",", fmt='%s')
        srcs, dsts = self.g.edges()
        srcdst = torch.column_stack((srcs, dsts))

        logs("Saving edges".format())
        np.savetxt(os.path.join(save_dir, "edges.csv"),
                   srcdst, delimiter=",", fmt='%s')

        if self.args.dataset == "ogbn-papers100M":
            logs("NUM_CLASSES: {self.num_classes}".format(**locals()))
        else:
            num_classes = len(torch.unique(labels))
            logs("NUM_CLASSES: {num_classes}".format(**locals()))

    def read_and_dump(self):
        self.read_dataset()
        self.dump()

    def ali_paths(self):
        node_table = os.path.join(self.args.ALI_DATA_REPO, "node_table")
        train_table = os.path.join(self.args.ALI_DATA_REPO, "train_table")
        valid_table = os.path.join(self.args.ALI_DATA_REPO, "valid_table")
        test_table = os.path.join(self.args.ALI_DATA_REPO, "test_table")
        edge_table = os.path.join(self.args.ALI_DATA_REPO, "edge_table")
        return node_table, train_table, valid_table, test_table, edge_table

    @timeit_factory()
    def dump_ali(self):
        srcs, dsts = self.g.edges()
        srcdst = torch.column_stack((srcs, dsts))
        feat = self.g.ndata['feat']
        nodes = self.g.nodes()
        labels = self.g.ndata['labels']
        Path(self.args.ALI_DATA_REPO).mkdir(parents=True, exist_ok=True)
        node_table, train_table, valid_table, test_table, edge_table = \
            self.ali_paths()

        for mode, save_dir in zip(
                self.args.cats.NTYPES, [train_table, valid_table, test_table]):
            with open(save_dir, 'w') as f:
                f.write("id:int64" + "\t" + "weight:float" + "\n")
                for node in tqdm(self.split_idx[mode].numpy()):
                    f.write(str(node) + "\t" + str(1.0) + "\n")

        srcdst_np = srcdst.numpy()
        with open(edge_table, 'w') as f:
            f.write("src_id: int64" + "\t"
                    + "dst_id: int64" + "\t"
                    + "weight: double" + "\n")
            for arr in tqdm(srcdst_np):
                f.write(str(arr[0]) + "\t" + str(arr[1]) + "\t" + "0.0" + "\n")
        
        num_nodes = len(nodes)
        with open(node_table, 'w') as f:
            f.write(
                "id:int64" +
                "\t" + "label:int64" + "\t" + "feature:string" + "\n")
            for i, (node_id, label, feat_arr) in tqdm(
                enumerate(
                    zip(nodes.numpy(), labels.numpy(), feat.numpy())),
                    total=num_nodes):
                total_string = '\t'.join(
                    [
                        str(node_id),
                        str(label),
                        ':'.join(np.char.mod('%f', feat_arr))]) + '\n'
                f.write(total_string)

        

        
