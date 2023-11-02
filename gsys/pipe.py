#!/local/env_dgl/bin/python -u

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
