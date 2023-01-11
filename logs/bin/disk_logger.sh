#!/usr/bin/env bash
if [ "$1" != "" ]; then
    LOG_DIR="$1"
    mkdir -p $LOG_DIR
else
    LOG_DIR="."
fi
LOG_FILENAME="$LOG_DIR/disk_$WORKER_NAME.log"
echo '' > $LOG_FILENAME
while true;
do
    datestr=`date "+%Y-%m-%d %H:%M:%S"`
    disk_log=$(iostat -dm 1 1 | sed '1,2d')
    echo -e "${datestr}\n${disk_log}" >> $LOG_FILENAME;
    sleep 1;
done




