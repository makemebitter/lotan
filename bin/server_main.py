import multiprocessing
from gsys.constants import network_constants
from gsys.ipc import Router
from gsys.ipc import ReceiverSender
from gsys.ipc import SyncServer
from gsys.ipc import Controller
from gsys.all_args import get_all_args
from gsys.utils import logsc

if __name__ == '__main__':
    args = get_all_args()
    manager = multiprocessing.Manager()
    event = manager.Event()
    event.set()  # We should keep running.
    stage = multiprocessing.Value('i', 0)
    incoming_mq = manager.Queue(
        maxsize=network_constants.INCOMING_MQ_MAX_SIZE)
    if args.omq_enque_test or args.omq_deque_test:
        OUTCOMING_MQ_MAX_SIZE = 250000
    else:
        OUTCOMING_MQ_MAX_SIZE = network_constants.OUTCOMING_MQ_MAX_SIZE
    outgoing_mq = manager.Queue(
        maxsize=OUTCOMING_MQ_MAX_SIZE)

    processes = []
    if args.worker_type == 'worker':
        from gsys.worker import Worker
    elif args.worker_type == 'prebatch_worker':
        if args.ipc_type == 'shm':
            from gsys.worker import PreBatchedWorkerSHM as Worker
        else:
            from gsys.worker import PreBatchedWorker as Worker
    elif args.worker_type == 'mpworker':
        from gsys.worker import MPDummyWorker as Worker

    worker = Worker(args, 'worker_0', stage=stage, event=event)
    worker_proc = worker.run(incoming_mq, outgoing_mq)
    processes.append(worker_proc)

    controller = Controller(args, 'controller_0', stage=stage, event=event)
    controller_proc = controller.run()
    processes.append(controller_proc)

    if args.omq_deque_test:
        # fully enqueue
        worker_proc.join()

    if args.network_arch == 'async':
        router = Router(args, 'router_0', stage=stage, event=event)
        router_proc = router.run()
        processes.append(router_proc)
        if args.omq_enque_test:
            # do nothing because do not deque
            pass
        else:
            if args.router_imq_link_test or args.omq_router_link_test or \
                    args.messenger_router_link_test:
                verbose = 0
            else:
                verbose = args.verbose
            receiver_senders = []
            for i in range(args.num_receiver_senders):
                receiver_sender = ReceiverSender(
                    args,
                    'receiver_sender_{}'.format(i),
                    stage=stage,
                    verbose=verbose, event=event)
                receiver_sender_proc = receiver_sender.run(
                    incoming_mq, outgoing_mq)
                receiver_senders.append(receiver_sender)
                processes.append(receiver_sender_proc)
            if args.omq_deque_test:
                with logsc("RECEIVER_SENDER PROCESSES", elapsed_time=True):
                    for receiver_sender in receiver_senders:
                        receiver_sender.proc.join()
    elif args.network_arch == 'sync':
        server = SyncServer(args, 'server_0', stage=stage, event=event)
        server_proc = server.run(incoming_mq, outgoing_mq)
        processes.append(server_proc)
    for proc in processes:
        proc.join()
