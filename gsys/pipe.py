#!/local/env_dgl/bin/python -u
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

from gsys.all_args import get_all_args
import os
import sys

if __name__ == '__main__':
    # total_in = 0
    # # make this buffered
    # for row in sys.stdin:
    #     # get row
    #     # send to PyT and wait
    #     # how to send to PyT
    #     # receives PyT response
    #     # print(new emb)
    #     total_in += 1
    #     print(row)
    #     # print(json.dumps(plain_sum(json.loads(row))))
    args = get_all_args(no_stdout=True)
    if not args.agg_pushdown:
        if args.io_type == 'raw_string':
            if args.ipc_type == 'socket':
                from gsys.ipc import \
                            MessengerRawStringPlain as Messenger
        else:
            raise NotImplementedError
    else:
        if args.io_type == 'raw_string':
            if args.ipc_type == 'shm':
                if args.sparse:
                    if args.first_sparse_pipe:
                        from gsys.ipc import \
                            MessengerRawStringSHMSparseFirstPipe as Messenger
                    else:
                        from gsys.ipc import \
                            MessengerRawStringSHMSparse as Messenger

                else:
                    from gsys.ipc import MessengerRawStringSHM as Messenger
            else:
                from gsys.ipc import MessengerRawString as Messenger
        elif args.io_type == 'byte':
            if args.ipc_type == 'shm':
                from gsys.ipc import MessengerByteSHM as Messenger
                # only shm is implemented
            else:
                raise NotImplementedError
        elif args.io_type == 'json':
            if args.network_arch == 'sync':
                from gsys.ipc import SyncMessenger as Messenger
            elif args.network_arch == 'async':
                from gsys.ipc import Messenger as Messenger
    messenger_ident = args.messenger_idx if \
        args.messenger_idx is not None else os.getpid()
    messenger = Messenger(
        args,
        "messenger_{}".format(messenger_ident),
        stage=0)
    print("{}, {}".format(messenger_ident, args),
          file=sys.stderr)
    messenger.main()
