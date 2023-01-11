

# 3-layer GCN on ogbn-arxiv datase
lotan_dataset="2" # ogbn-arxiv
xavier="True"
leaky="True"
batchnorm="False"
model="gcn"
epoch=10
COMMON_OPTIONS="--lotan_model_batching --drill_down_mb_size 8 --drill_down_mb_below_batching_start 0 --model ${model} --model_batchnorm ${batchnorm} --model_xavier ${xavier} --model_leaky ${leaky}"
export num_layers=3
bash run_mb.sh "" "$epoch" "" "${COMMON_OPTIONS}" "$num_layers"




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