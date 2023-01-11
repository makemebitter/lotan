#!/usr/bin/env bash
if [ "$1" != "" ]; then
    LOG_DIR="$1"
    mkdir -p $LOG_DIR
else
    LOG_DIR="."
fi

LOG_FILENAME="$LOG_DIR/cpu_utilization_$WORKER_NAME.log"

while true;
do
    datestr=`date "+%Y-%m-%d %H:%M:%S"`
    cpu="$[100-$(vmstat 1 2|tail -1|awk '{print $15}')]%"
    mem=$(free | grep Mem | awk '{print $3/$2 * 100.0 "%"}')
    echo -e "${datestr}\n${cpu},${mem}" >> $LOG_FILENAME;
    sleep 1;
done