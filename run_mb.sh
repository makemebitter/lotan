#!/usr/bin/env bash
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

source runner_helper.sh
num_layers=${5:-"3"}
lotan_dataset=${6:-"2"}
aggPushDown=${7:-"1"}
noReverseGraph=${8:-"0"}
ioType=${9:-"1"}
ipcType=${10:-"1"}
E2D=${11:-"0"}
normalize=${12:-"1"}
sparse=${13:-"0"}

# dump=${8:-"0"}

if [ "$lotan_dataset" -eq "1" ]
then
   dataset="ogbn-products"
elif [ "$lotan_dataset" -eq "2" ]
then
   dataset="ogbn-arxiv"
elif [ "$lotan_dataset" -eq "3" ]
then
   dataset="ogbn-papers100M"
fi

echo $OPTIONS
echo $num_layers
echo $dataset

# spark_worker_cores=40
# echo "Overwriting SPARK CONFIGS"
# $PARALLEL_SSH_ALL "cd /mnt/nfs/gsys; \"$DGL_PY\" overwrite_spark_env.py --spark_worker_cores ${spark_worker_cores}"
# echo "Restarting SPARK"
RESTART_SPARK

cd graphp
sbt assembly
cd ..
CMD_STR='spark-submit --conf "spark.executor.extraJavaOptions=-Ddev.ludovic.netlib.blas.nativeLibPath=/usr/lib/x86_64-linux-gnu/mkl/liblapack.so -Ddev.ludovic.netlib.lapack.nativeLibPath=/usr/lib/x86_64-linux-gnu/mkl/liblapack.so"  --class "org.apache.spark.graphx.lotan.main.SimpleApp" graphp/target/scala-2.12/simple-project_2.12-1.0.jar'

# -----------------------------------------------------------------------------

epochs=$EPOCHS


export EXP_NAME="spark_dump"
$DGL_PY dgl_to_spark_data.py --self_loop --undirected --dataset ${dataset} 
# -------------------------------------MB-------------------------------------
# --lotan_model_batching
options="--io_type byte --hard_partition $OPTIONS"
numEParts=80
numVParts=80
# MASTER="master"
# -----------------------------------server-----------------------------------
export EXP_NAME="server"
MAKE_CLIENT_LOG_DIR
# some dummy configs, deprecated, true configuration is in gsys/constants.py
lr=0.01
dropout=0.0
optimizer="adam"

$PARALLEL_SSH_ALL "export DGL_PY=$DGL_PY; cd $LOTAN_NFS_ROOT/bin; bash run_server_main.sh \"$options --master ${MASTER_IP} --model_num_layers ${num_layers} --dataset ${dataset} --model_lr ${lr} --model_epochs ${epochs} --save_model_root ${MODEL_DIR} --size ${SIZE} --model_dropout ${dropout} --model_optimizer ${optimizer}\" 2>&1 | tee -a ${LOG_DIR}/${EXP_NAME}/"'$WORKER_NAME.log' &
echo "Use the following command to monitor the DL Engine"
echo "------------------------------------------------------------------------"
echo "tail -f ${LOG_DIR}/${EXP_NAME}/"'$WORKER_NAME.log'
echo "------------------------------------------------------------------------"
# ----------------------------------------------------------------------------


# -----------------------------------graphx-----------------------------------
numMachines=$SIZE
# CPUs=${spark_worker_cores}
export EXP_NAME="graphx"
# --drillDown 1

SPARK_BASE_CMD="$CMD_STR --E2D ${E2D} --numVParts ${numVParts} --ipcType ${ipcType} --ioType ${ioType} --noReverseGraph ${noReverseGraph} --drillDown 1 --fillFeatures 1 --normalize ${normalize} --sparse ${sparse} --run 1 --numMachines ${numMachines} --numEParts ${numEParts} --numEpochs ${epochs} --dataset ${lotan_dataset} --miniBatchSize 500000 --savePlain 0 --aggPushDown ${aggPushDown} --numLayers ${num_layers}"
# ----------------------------------------------------------------------------
# ----------------------------hardPartition-----------------------------------

set +e
$HADOOP_HOME/bin/hdfs dfs -rm -r /edgesRev
$HADOOP_HOME/bin/hdfs dfs -rm -r /edges
$HADOOP_HOME/bin/hdfs dfs -rm -r /vertices
set -e

EXECUTE_CMD="$SPARK_BASE_CMD --hardPartition 1"
RUN_EXP "$EXECUTE_CMD"
echo "Dumped graph to HDFS"

EXECUTE_CMD="$SPARK_BASE_CMD --hardPartitionRead 1"
RUN_EXP "$EXECUTE_CMD"

# ----------------------------------------------------------------------------
set +e 
$PARALLEL_SSH_ALL "pkill -f server_main.py; pkill -f pipe.py"
# ----------------------------------------------------------------------------
touch $LOG_DIR/"${OPTIONS}_${numEParts}_${num_layers}"
