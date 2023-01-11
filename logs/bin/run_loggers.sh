#!/usr/bin/env bash
NFS_DIR=${1:-"/mnt/nfs"}
GPU_WORKERS=$2
duty=$3
datestr=`date "+%Y_%m_%d"`
NETWORK_LOG_DIR=$NFS_DIR/logs/${datestr}/network_logs
DISK_LOG_DIR=$NFS_DIR/logs/${datestr}/disk_logs
CPU_LOG_DIR=$NFS_DIR/logs/${datestr}/cpu_logs
GPU_LOG_DIR=$NFS_DIR/logs/${datestr}/gpu_logs
__dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
nohup bash ${__dir}/network_logger.sh $NETWORK_LOG_DIR eno1 enp94s0f0 &
nohup bash ${__dir}/disk_logger.sh $DISK_LOG_DIR &
nohup bash ${__dir}/cpu_logger.sh $CPU_LOG_DIR &

if [ $GPU_ENABLED -eq 1 ]; then
nohup bash ${__dir}/gpu_logger.sh $GPU_LOG_DIR &
fi
