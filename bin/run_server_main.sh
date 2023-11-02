#!/usr/bin/env bash
OPTIONS=${1:-""}
agg_pushdown=${2:-"True"}
ipc_type=${3:-"shm"}

pkill -f 'pipe.py' && sleep 2
pkill -f 'server_main.py' && sleep 2
$DGL_PY -u server_main.py --worker_type "prebatch_worker" --verbose 0 --gpu --agg_pushdown ${agg_pushdown} --ipc_type ${ipc_type} $OPTIONS
