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

# 3-layer GCN on ogbn-arxiv datase
lotan_dataset="2" # ogbn-arxiv
xavier="True"
leaky="True"
batchnorm="False"
model="gcn" # or "gin"
size="1" # number of computational nodes
export numEParts=40 # number of edge partitions
export numVParts=40 # number of vertex partitions
epoch=10
# to enable model batching, change drill_down_mb_size to a higher number
COMMON_OPTIONS="--lotan_model_batching --drill_down_mb_size 1 --drill_down_mb_below_batching_start 0 --model ${model} --model_batchnorm ${batchnorm} --model_xavier ${xavier} --model_leaky ${leaky}"
export num_layers=3
bash run_mb.sh "" "$epoch" "$size" "${COMMON_OPTIONS}" "$num_layers" "$lotan_dataset"




# # 
# # 
# # Ablation study
# # naive
# # -----------------------------------------------------------------------------
# xavier="True"
# leaky="True"
# batchnorm="False"
# model="gcn"
# export num_layers=3
# lotan_dataset="2"


# COMMON_OPTIONS="--model ${model} --model_batchnorm ${batchnorm} --model_xavier ${xavier} --model_leaky ${leaky}"
# num_epochs="200"

# # naive
# FIN_OPTIONS="${COMMON_OPTIONS} --agg_pushdown False --io_type raw_string --ipc_type socket"
# aggPushDown="0"
# noReverseGraph="1"
# ioType="0"
# ipcType="0"
# E2D="3"
# bash run_mb.sh "" "${num_epochs}" "" "${FIN_OPTIONS}" "$num_layers" "$lotan_dataset" "$aggPushDown" "$noReverseGraph" "$ioType" "$ipcType" "$E2D"

# # # + ReverseG
# FIN_OPTIONS="${COMMON_OPTIONS} --agg_pushdown False --io_type raw_string --ipc_type socket"
# aggPushDown="0"
# noReverseGraph="0"
# ioType="0"
# ipcType="0"
# E2D="0"
# bash run_mb.sh "" "${num_epochs}" "" "${FIN_OPTIONS}" "$num_layers" "$lotan_dataset" "$aggPushDown" "$noReverseGraph" "$ioType" "$ipcType" "$E2D"

# # # + AggPushDown (StringIO version at this moment)
# FIN_OPTIONS="${COMMON_OPTIONS} --agg_pushdown True --io_type raw_string --ipc_type socket"
# aggPushDown="1"
# noReverseGraph="0"
# ioType="0"
# ipcType="0"
# E2D="0"
# bash run_mb.sh "" "${num_epochs}" "" "${FIN_OPTIONS}" "$num_layers" "$lotan_dataset" "$aggPushDown" "$noReverseGraph" "$ioType" "$ipcType" "$E2D"

# # # + Messenger optimizations

# FIN_OPTIONS="${COMMON_OPTIONS} --agg_pushdown True --io_type byte --ipc_type shm"
# aggPushDown="1"
# noReverseGraph="0"
# ioType="1"
# ipcType="1"
# E2D="0"
# bash run_mb.sh "" "${num_epochs}" "" "${FIN_OPTIONS}" "$num_layers" "$lotan_dataset" "$aggPushDown" "$noReverseGraph" "$ioType" "$ipcType" "$E2D"

# # # + Model batching
# COMMON_OPTIONS="--lotan_model_batching --drill_down_mb_size 8 --drill_down_mb_below_batching_start 0 --model ${model} --model_batchnorm ${batchnorm} --model_xavier ${xavier} --model_leaky ${leaky}"
# FIN_OPTIONS="${COMMON_OPTIONS} --agg_pushdown True --io_type byte --ipc_type shm"
# aggPushDown="1"
# noReverseGraph="0"
# ioType="1"
# ipcType="1"
# E2D="0"
# bash run_mb.sh "" "${num_epochs}" "" "${FIN_OPTIONS}" "$num_layers" "$lotan_dataset" "$aggPushDown" "$noReverseGraph" "$ioType" "$ipcType" "$E2D"
# # -----------------------------------------------------------------------------