#!/usr/bin/env bash
if [ "$1" != "" ]; then
    LOG_DIR="$1"
    mkdir -p $LOG_DIR
else
    LOG_DIR="."
fi

LOG_FILENAME="$LOG_DIR/gpu_utilization_$WORKER_NAME.log"

while true;
do  
    datestr=`date "+%Y-%m-%d %H:%M:%S"`
    gpu_uti=$(nvidia-smi --query-gpu=utilization.gpu,utilization.memory,power.draw --format=csv,noheader)
    echo -e "${datestr}\n${gpu_uti}" >> $LOG_FILENAME
    sleep 1;
done