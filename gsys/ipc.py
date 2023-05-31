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

import os
import errno
import pickle
from .constants import constants
from .constants import network_constants
from .messages import messages
from .utils import logs
import zmq
import multiprocessing
from multiprocessing import shared_memory
from threading import Thread
from queue import Full
from queue import Empty
import json
import datetime
import numpy as np
import sys
import traceback

SOCKET_NP_LONG_TYPE = np.dtype(np.compat.long).newbyteorder('>')
SOCKET_NP_FLOAT_TYPE = np.dtype(np.float32).newbyteorder('>')
NATIVE_NP_LONG_TYPE = np.dtype(np.compat.long).newbyteorder('=')
NATIVE_NP_FLOAT_TYPE = np.dtype(np.float32).newbyteorder('=')

CHAR_MOD = '%e'


def get_fifo_m2w(i):
    return get_fifo(i, constants.M2W)


def get_fifo_w2m(i):
    return get_fifo(i, constants.W2M)


def get_fifo(i, mode):
    return os.path.join(constants.PIPES_ROOT, '_'.join([mode, str(i)]))


class NamedPipes(object):
    def __init__(self, args):
        self.args = args

    def init(self):
        for i in range(self.args.concurrency):
            #  make all the pipes
            try:
                os.mkfifo(get_fifo_m2w(i))
                os.mkfifo(get_fifo_w2m(i))
            except OSError as oe:
                if oe.errno != errno.EEXIST:
                    raise


def serialize(objs):
    return pickle.dumps(objs, protocol=-1)


def deserialize(msg):
    return pickle.loads(msg)


class Socket(object):
    def __init__(self, context, socket_type, ident, connect, addr):
        self.context = context
        self.context.setsockopt(
            zmq.IO_THREADS, network_constants.ZMQ_IO_THREADS)
        self.socket_type = socket_type
        self.ident = ident
        self.conenct = connect
        self.addr = addr
        self.socket = self.context.socket(self.socket_type)
        self.socket.setsockopt_string(zmq.IDENTITY, ident)
        self.socket.setsockopt(zmq.RCVHWM, network_constants.ZMQ_RCVHWM)
        self.socket.setsockopt(zmq.SNDHWM, network_constants.ZMQ_SNDHWM)

        if socket_type == zmq.ROUTER:
            self.socket.setsockopt(zmq.ROUTER_MANDATORY, 1)

        if connect == 'connect':
            self.socket.connect(addr)
        elif connect == 'bind':
            self.socket.bind(addr)

        self.pollin = zmq.Poller()
        self.pollin.register(self.socket, zmq.POLLIN)

        self.pollout = zmq.Poller()
        self.pollout.register(self.socket, zmq.POLLOUT)

    def is_pollin(self, timeout=0):
        return self.is_poll(timeout, 'in')

    def is_pollout(self, timeout=0):
        return self.is_poll(timeout, 'out')

    def is_poll(self, timeout=0, which='in'):

        if which == 'in':
            poll = self.pollin
        else:
            poll = self.pollout
        sockets = dict(poll.poll(timeout=timeout))
        return self.socket in sockets

    def send_obj(self, obj):
        # blocking send
        self.socket.send(serialize(obj))

    def recv_obj(self):
        # blocking recv
        return deserialize(self.socket.recv())

    def recv_multipart(self, *args, **kwargs):
        return self.socket.recv_multipart(*args, **kwargs)

    def send_multipart(self, *args, **kwargs):
        return self.socket.send_multipart(*args, **kwargs)

    def recv(self, *args, **kwargs):
        return self.socket.recv(*args, **kwargs)

    def send(self, *args, **kwargs):
        return self.socket.send(*args, **kwargs)


class AsyncSocket(Socket):
    pass


class SyncSocket(Socket):
    pass


class IPCBase(object):
    def __init__(
            self, args, ident, stage, no_stdout=False, verbose=0, event=None):
        self.args = args
        self.ident = ident
        self.stage = stage
        self.no_stdout = no_stdout
        self.verbose = verbose
        self.event = event

    def run(self, *args):
        proc = multiprocessing.Process(target=self.main, args=args)
        proc.start()
        self.proc = proc
        # if not self.no_stdout:
        logs("{} launched".format(self.ident))
        return proc


class Router(IPCBase):
    def main(self):
        # context = zmq.Context()
        # frontend = AsyncSocket(
        #     context,
        #     zmq.ROUTER,
        #     '{}_frontend'.format(self.ident),
        #     'bind',
        #     network_constants.ROUTER_ADDRESS)
        # backend = AsyncSocket(
        #     context,
        #     zmq.DEALER,
        #     '{}_backend'.format(self.ident),
        #     'bind',
        #     network_constants.ROUTER_BACKEND_IPC)
        # zmq.proxy(frontend.socket, backend.socket)
        pass


class Controller(IPCBase):
    def main(self):
        self.first_layer_finished = False
        context = zmq.Context()
        socket = SyncSocket(context, zmq.REP, self.ident,
                            'bind', network_constants.CONTROLLER_ADDRESS)

        # ROUTER ###############
        frontend = AsyncSocket(
            context,
            zmq.ROUTER,
            '{}_frontend'.format(self.ident),
            'bind',
            network_constants.ROUTER_ADDRESS)
        backend = AsyncSocket(
            context,
            zmq.DEALER,
            '{}_backend'.format(self.ident),
            'bind',
            network_constants.ROUTER_BACKEND_IPC)
        proxy_thread = Thread(
            target=lambda: zmq.proxy(
                frontend.socket, backend.socket))
        proxy_thread.daemon = True
        proxy_thread.start()

        ####################################
        while self.event.is_set():
            msg = socket.recv_obj()
            if msg == messages.LAYER_FINISHED:
                socket.send_obj(messages.CONTROLLER_ACKNOWLEDGE)
                logs(msg)
                self.first_layer_finished = True
                self.stage.value = 1
            elif msg == messages.TERM:
                logs(msg)
                # shut down
                self.event.clear()
                break
            else:
                raise Exception("Wrong messages: {msg}".format(**locals()))


class ReceiverSender(IPCBase):

    def state_str(self):
        return "received: {}, put: {}, sent: {}, polled: {}".format(
            self.received, self.put, self.sent, self.polled)

    def report_throughput(self, last, processed, report_freq):
        if processed % report_freq == 0:
            if last is None:
                last = datetime.datetime.now()
            else:
                logs("{}: PROCESSED {}!, {}".format(
                    report_freq,
                    self.ident, self.state_str()))
                now = datetime.datetime.now()
                elapsed = now - last
                throughput = processed / elapsed.total_seconds()
                logs("{self.ident}, THROUGHPUT: {throughput}".format(
                    **locals()))
        return last

    def main(self, incoming_mq, outgoing_mq):
        context = zmq.Context()
        socket = AsyncSocket(context, zmq.DEALER, self.ident,
                             'connect', network_constants.ROUTER_BACKEND_IPC)

        # worker = context.socket(zmq.DEALER)
        # worker.connect(network_constants.ROUTER_BACKEND_IPC)
        # poll = zmq.Poller()
        # poll.register(worker, zmq.POLLIN)
        to_imq = None
        to_socket = None
        self.received = 0
        self.sent = 0
        self.put = 0
        self.polled = 0
        receive_turn = True
        last = None
        if self.args.router_messenger_link_test \
                or self.args.router_imq_link_test:
            with open('/mnt/nfs/gsys/msg_out_example.pickle', 'rb') as f:
                msg = f.read()
                # json_obj = pickle.loads(msg)

        if self.args.router_imq_link_test:
            to_imq = ("dummmy_ident", msg)

        if self.args.router_messenger_link_test:
            with open('/mnt/nfs/gsys/msg_out_example.pickle', 'rb') as f:
                dummy_msg = f.read()
            online = 0
            # wait for all to come online
            address_book = []
            while online < 40:
                ident, msg = socket.recv_multipart()
                obj = pickle.loads(msg)
                if obj != "Hi":
                    raise Exception("Not handshake message: {}".format(obj))
                address_book.append(ident)
                online += 1
                logs("Online: {}, {}/40".format(ident, len(address_book)))
            for ident in address_book:
                obj = "Go"
                msg = pickle.dumps(obj)
                socket.send_multipart([ident, msg])
                logs("Handshake sent to {}".format(ident))
            for i in range(250000):
                logs("Sending")
                # round robin send
                ith = i % 40
                ident = address_book[ith]
                socket.send_multipart([ident, dummy_msg])
                logs("Sent to: {}, {}/250000".format(ident, i, ))
            for ident in address_book:
                obj = "End"
                msg = pickle.dumps(obj)
                socket.send_multipart([ident, msg])
                logs("Endshake sent to {}".format(ident))
        else:
            while self.event.is_set():
                if receive_turn is True:
                    if to_imq is None:
                        if self.args.router_imq_link_test:
                            to_imq = ("dummmy_ident", msg)
                        elif socket.is_pollin():
                            # receive non-blocking
                            ident, msg = socket.recv_multipart()
                            self.received += 1
                            if self.verbose > 1:
                                logs(
                                    'RouterSender received from {}'.format(
                                        ident))
                            if self.verbose > 0:
                                logs(self.state_str())
                            to_imq = (ident, msg)
                    else:
                        if self.args.messenger_router_link_test:
                            # discard the message and don't do enqueue
                            to_imq = None
                            last = self.report_throughput(
                                last, processed=self.received, report_freq=500)
                        else:
                            # enque non-blocking
                            try:
                                # logs('RouterSender Putting {}
                                # into the imq'.format(to_imq))
                                incoming_mq.put(to_imq, block=False)
                                to_imq = None
                                self.put += 1
                            except Full:
                                if self.verbose > 1:
                                    logs("IMQ FULL!, {}".format(
                                        self.state_str()))

                    receive_turn = False
                else:
                    if to_socket is None:
                        # deque non-blocking
                        try:
                            (ident, msg) = outgoing_mq.get(timeout=0)
                            self.polled += 1
                            if self.args.omq_router_link_test \
                                    or self.args.omq_deque_test:
                                to_socket = None
                                report_freq = 500
                                last = self.report_throughput(
                                    last,
                                    processed=self.polled,
                                    report_freq=report_freq)

                            else:
                                to_socket = (ident, msg)
                        except Empty:
                            # nothing happens
                            if self.verbose > 1:
                                logs("OMQ EMPTY!, {}".format(self.state_str()))
                            if self.args.omq_deque_test:
                                # all popped
                                break

                    else:
                        try:
                            # send message non-blocking
                            if socket.is_pollout():
                                (ident, msg) = to_socket
                                socket.send_multipart([ident, msg])
                                self.sent += 1
                                to_socket = None
                                if self.verbose > 1:
                                    logs(
                                        'RouterSender sent to {}'.format(
                                            ident))
                                if self.verbose > 0:
                                    logs(self.state_str())
                        except zmq.Again:
                            logs("SEND Failure!")
                    receive_turn = True


class DummyMessenger(IPCBase):
    dummy_pipeout = """{"_1":16819,"_2":[{"_1":5920,"_2":[1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0]},{"_1":12928,"_2":[1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0]},{"_1":18554,"_2":[1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0]},{"_1":18606,"_2":[1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0]},{"_1":5384,"_2":[1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0]},{"_1":2409,"_2":[1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0]},{"_1":7369,"_2":[1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0]}]}
    """
    dummy_object = json.loads(dummy_pipeout)

    def main(self):
        context = zmq.Context()
        socket = AsyncSocket(
            context,
            zmq.DEALER,
            self.ident,
            'connect',
            network_constants.ROUTER_ADDRESS)
        sent = 0
        received = 0

        for j in range(10):
            socket.send_obj(self.dummy_object)
            if not self.no_stdout:
                logs("{} just sent a message".format(self.ident))
            sent += 1
            if socket.is_pollin():
                obj = socket.recv_obj()
                received += 1
                if not self.no_stdout:
                    logs((self.ident, j, obj))
                    logs("Sent: {}, Received: {}".format(sent, received))
        # read all left
        while received < sent:
            obj = socket.recv_obj()
            received += 1
            if not self.no_stdout:
                logs((self.ident, j, obj))
                logs("Sent: {}, Received: {}".format(sent, received))


class DatumBase(object):

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        length = len(self.unique_vs)
        keys = self.unique_vs[:5]
        return "Name: {self.name}| Size: {length}| Keys: {keys} ..."\
            .format(**locals())


class Datum(DatumBase):
    name = "Datum"
    shm = False

    def __init__(self, H_u, V, U, unique_vs):
        self.H_u = H_u
        self.V = V
        self.unique_vs = unique_vs
        self.U = U
        self.batch_indices = np.array(
            self.unique_vs, dtype=NATIVE_NP_LONG_TYPE)


class DatumSingleVector(DatumBase):
    name = "DatumSingleVector"
    shm = False

    def __init__(self, grads, unique_vs):
        self.grads = grads
        self.unique_vs = unique_vs
        self.batch_indices = np.array(
            self.unique_vs, dtype=NATIVE_NP_LONG_TYPE)


class DatumSHM(Datum):
    # row-based datum, has a shm-based numpy array
    name = "DatumSHM"
    shm = True

    def __init__(self, shm_name, shm_name_ids, shape, shape_ids, *args, **kwargs):
        self.shm_name = shm_name
        self.shm_name_ids = shm_name_ids
        self.shape = shape
        self.shape_ids = shape_ids
        super(DatumSHM, self).__init__(*args, **kwargs)

    def __str__(self):
        return "Name: {self.name}, SHM: {self.shm_name}".format(**locals())

    def unload(self):
        self.existing_shm = shared_memory.SharedMemory(
            name=self.shm_name)
        self.existing_shm_ids = shared_memory.SharedMemory(
            name=self.shm_name_ids)
        self.shm_arr = np.ndarray(
            self.shape,
            dtype=NATIVE_NP_FLOAT_TYPE,
            buffer=self.existing_shm.buf)
        self.shm_arr_ids = np.ndarray(
            self.shape_ids,
            dtype=NATIVE_NP_LONG_TYPE,
            buffer=self.existing_shm_ids.buf)
        # print("Datum Unloaded: {}, {}".format(
        #     self.shm_arr, self.shm_arr.dtype.byteorder), file=sys.stderr)
        return self.shm_arr, self.shm_arr_ids

    def populate(self):
        # column
        self.H_u = self.shm_arr[:, self.H_u:]
        # self.unique_vs = self.shm_arr[:, self.unique_vs]
        self.unique_vs = self.shm_arr_ids.astype(NATIVE_NP_LONG_TYPE)
        self.batch_indices = self.unique_vs.astype(NATIVE_NP_LONG_TYPE)
        if self.V is not None:
            # V needs to be copied
            self.V = self.shm_arr[:, self.V].copy()
        # other fields haven't been implemented
        self.U = None

    def close(self):
        self.existing_shm.close()
        self.existing_shm_ids.close()

    def unlink(self):
        try:
            self.existing_shm.unlink()

        except Exception as e:
            print("Exception happend during unlinking")
            print(e)
            traceback.print_exc()

        try:
            self.existing_shm_ids.unlink()

        except Exception as e:
            print("Exception happend during unlinking")
            print(e)
            traceback.print_exc()

    def clean(self):
        self.close()
        self.unlink()
        print("Cleaned SHM: {}".format(self.shm_name), file=sys.stderr)
        print("Cleaned SHM: {}".format(self.shm_name_ids), file=sys.stderr)


class DatumSingleVectorSHM(DatumSHM):
    name = "DatumSingleVectorSHM"
    # yeah inheritance is messed up here, don't blame me

    def __init__(self, shm_name, shm_name_ids, shape, shape_ids, grads, unique_vs):
        self.shm_name = shm_name
        self.shm_name_ids = shm_name_ids
        self.shape = shape
        self.shape_ids = shape_ids
        self.grads = grads
        self.unique_vs = unique_vs

    def populate(self):
        # column
        self.grads = self.shm_arr[:, self.grads:]
        self.unique_vs = self.shm_arr_ids
        self.batch_indices = self.unique_vs.astype(NATIVE_NP_LONG_TYPE)


class MessengerBase(IPCBase):
    # def __init__(self, *args, **kwargs):
    #     super(Messenger, self).__init__(*args, **kwargs)

    def zmq_init(self):
        context = zmq.Context()
        # data socket
        socket = AsyncSocket(
            context,
            zmq.DEALER,
            self.ident,
            'connect',
            network_constants.ROUTER_ADDRESS)

        ctl_socket = SyncSocket(
            context,
            zmq.REQ,
            self.ident,
            'connect',
            network_constants.CONTROLLER_ADDRESS)
        self.context = context
        self.socket = socket
        self.ctl_socket = ctl_socket
        return context, socket, ctl_socket

    # def json_objs_to_mini_batch_single_vector_lift_json(self, objs):
    #     new_dict = {}
    #     for x in objs:
    #         new_dict[x['_1']] = x['_2']
    #     unique_vs = []
    #     grads = []
    #     for v in sorted(list(new_dict.keys())):
    #         grad = new_dict[v]
    #         unique_vs.append(v)
    #         grads.append(grad)
    #     grads = np.asarray(grads)
    #     return DatumSingleVector(grads=grads, unique_vs=unique_vs)

    # def json_objs_to_mini_batch_lift_json(self, objs):
    #     new_dict = {}
    #     for x in objs:
    #         new_dict[x['_1']] = x['_2']
    #     H_u = []
    #     V = []
    #     unique_vs = []
    #     U = []
    #     for v in sorted(list(new_dict.keys())):
    #         h_us = new_dict[v]
    #         unique_vs.append(v)
    #         for h_u_dict in h_us:
    #             u = h_u_dict['_1']
    #             h_u = h_u_dict['_2']
    #             H_u.append(h_u)
    #             U.append(u)
    #             V.append(v)

    #     H_u = np.asarray(H_u)
    #     # V = np.asarray(V)
    #     # U = np.asarray(U)
    #     return Datum(H_u=H_u, V=V, U=U, unique_vs=unique_vs)

    def objs_to_mini_batch_single_vector(self, objs):
        unique_vs = []
        vectors = []
        for obj in objs:
            v, vector = obj
            unique_vs.append(v)
            vectors.append(vector)
        # vectors = np.asarray(vectors)
        if self.args.messenger_single_vector and \
                not self.args.messenger_backprop:
            return Datum(H_u=vectors, V=None, U=None, unique_vs=unique_vs)
        else:
            return DatumSingleVector(grads=vectors, unique_vs=unique_vs)

    # this is for json

    def objs_to_mini_batch(self, objs):
        H_u = []
        V = []
        unique_vs = []
        U = []
        for obj in objs:
            v, h_us = obj
            unique_vs.append(v)
            for u, h_u in h_us:
                H_u.append(h_u)
                U.append(u)
                V.append(v)
        # H_u = np.asarray(H_u)

        return Datum(H_u=H_u, V=V, U=U, unique_vs=unique_vs)

    def batch_gen(self, mini_batch_size=5000):
        if self.args.messenger_single_vector \
            or self.args.messenger_final_backprop \
                or self.args.messenger_backprop:
            parse_func = self.objs_to_mini_batch_single_vector
        else:
            parse_func = self.objs_to_mini_batch
        return self.batch_gen_(parse_func, mini_batch_size)

    def vertex_gen(self):
        for string in sys.stdin:
            yield self.read_sys(string)

    def write_sys_minibatch(self, objs):
        for obj in objs:
            self.write_sys(obj)

    def batch_gen_(self, parse_func, mini_batch_size=5000):
        objs = []
        curr = 0
        self.num_batches = 0
        for obj in self.vertex_gen():
            objs.append(obj)
            curr += 1
            if mini_batch_size == curr:
                yield parse_func(objs)
                self.num_batches += 1
                objs = []
                curr = 0
        # flush rest
        if objs:
            yield parse_func(objs)
            self.num_batches += 1

    def write_dummy(self):
        print("")

    def final_write(self):
        print(messages.ALL_FINISHED)

    def main(self):
        import sys

        self.zmq_init()
        sent = 0
        received = 0

        if self.args.worker_throughput_test:
            for json_string in sys.stdin:

                print(json_string)
                # obj = self.read_sys(json_string)
                # msg = pickle.dumps(obj)
                # json_obj = pickle.loads(msg)
                # self.write_sys(json_obj)

        elif self.args.messenger_router_link_test or \
                self.args.router_imq_link_test:
            with open('/mnt/nfs/gsys/msg_in_example.pickle', 'rb') as f:
                msg = f.read()
            json_obj = pickle.loads(msg)
            for _ in range(6250):
                self.socket.send_obj(json_obj)

            # notify graphx to stop
            print(" ")
        elif self.args.router_messenger_link_test:
            # send address to router
            self.socket.send_obj("Hi")
            # wait for commence message
            commence = self.socket.recv_obj()
            if commence != 'Go':
                raise Exception("Received not right handshake: {}".format(
                    commence))
            while True:
                # just read
                obj = self.socket.recv_obj()
                if obj == 'End':
                    break
            print(" ")

        elif self.args.send_finished:
            self.ctl_socket.send_obj(messages.LAYER_FINISHED)
            msg = self.ctl_socket.recv_obj()
            if msg == messages.CONTROLLER_ACKNOWLEDGE:
                print(msg)
            else:
                raise Exception("Wrong message: {msg}".format(**locals()))
            # for string in sys.stdin:
            #     string = string
            #     break
            # if int(string) % self.args.cpu_cores_per_machine == 0:

            # else:
            #     print(" ")
        elif self.args.send_term:
            self.ctl_socket.send_obj(messages.TERM)
            print('')

        else:
            # actual IPC code
            if self.args.worker_type == 'prebatch_worker':
                gen = self.batch_gen(mini_batch_size=self.args.mini_batch_size)
                write = self.write_sys_minibatch
            else:
                gen = self.vertex_gen()
                write = self.write_sys

            if self.args.messenger_final_backprop:
                # send everything
                for obj in gen:
                    self.socket.send_obj(obj)
                    self.write_dummy()
                # recieive the only reply
                _ = self.socket.recv_obj()
                self.final_write()

            else:
                for obj in gen:
                    self.socket.send_obj(obj)
                    sent += 1
                    # if socket.is_pollin():
                    obj = self.socket.recv_obj()
                    # print(obj, file=sys.stderr)
                    received += 1
                    write(obj)
                # # read all left
                # while received < sent:
                #     obj = self.socket.recv_obj()
                #     # print(obj, file=sys.stderr)
                #     received += 1
                #     write(obj)


class SyncServer(IPCBase):
    def main(self, incoming_mq, outgoing_mq):
        context = zmq.Context()
        socket = SyncSocket(
            context,
            zmq.REP,
            self.ident,
            'bind',
            network_constants.ROUTER_ADDRESS)
        while True:
            msg = socket.recv()
            incoming_mq.put(('dummy_ident', msg))
            _, msg = outgoing_mq.get()
            socket.send(msg)


class SyncMessenger(MessengerBase):
    def main(self):
        import sys
        context = zmq.Context()
        socket = SyncSocket(
            context,
            zmq.REQ,
            self.ident,
            'connect',
            network_constants.ROUTER_ADDRESS)
        sent = 0
        received = 0
        # one send one recv and all blocking
        for json_string in sys.stdin:
            socket.send_obj(self.read_sys(json_string))
            sent += 1
            json_obj = socket.recv_obj()
            received += 1
            self.write_sys(json_obj)


class Messenger(MessengerBase):

    def read_sys(self, json_string):
        return json.loads(json_string)

    def write_sys(self, json_obj):
        v, vector = json_obj
        new_obj = (v, vector.tolist())
        print(json.dumps(new_obj))


class MessengerRawString(MessengerBase):
    """Single vector in, single vector out"""

    def read_sys(self, raw_string):
        splitted = raw_string.split(',')
        v = int(splitted[0])
        vector = np.asarray(splitted[1:], dtype=NATIVE_NP_FLOAT_TYPE)
        return (v, vector)

    def write_sys(self, obj):
        v, vector = obj
        # x_arrstr = np.char.mod('%f', vector)
        x_arrstr = np.char.mod(CHAR_MOD, vector)
        string = str(v) + "," + ','.join(x_arrstr)
        print(string)


class MessengerRawStringPlain(MessengerRawString):
    """Forward: Gather array in, single emb out
    Backward: Grad in, Map of grads out
    """

    def write_sys(self, *args, **kwargs):
        # forward
        if not self.args.messenger_backprop and not self.args.messenger_plain_forbackward:
            return super(MessengerRawStringPlain, self).write_sys(*args, **kwargs)
        # Backward
        else:
            return self.write_sys_map_gradients(*args, **kwargs)

    def read_sys(self, *args, **kwargs):
        # forward
        if not self.args.messenger_backprop:
            return self.read_sys_gather_array(*args, **kwargs)
        # Backward
        else:
            return super(MessengerRawStringPlain, self).read_sys(*args, **kwargs)

    def write_sys_map_gradients(self, obj):
        res = ""
        v, payload = obj
        res += str(int(v))
        res += ","
        res += str(int(len(payload)))

        for u, h_u_grad in payload:
            res += ","
            res += str(int(u))
            res += ","
            x_arrstr = np.char.mod(CHAR_MOD, h_u_grad)
            res += ','.join(x_arrstr)
        print(res)

    def read_sys_gather_array(self, raw_string):
        splitted = raw_string.split(',')
        v = int(splitted[0])
        length_of_collection = int(splitted[1])
        total_size_of_collection = len(splitted) - 2
        each_length = total_size_of_collection // length_of_collection

        matrix = np.asarray(splitted[2:], dtype=NATIVE_NP_FLOAT_TYPE).reshape(
            (length_of_collection, each_length))
        h_us = [(int(row[0]), row[1:]) for row in matrix]
        return (v, h_us)


class SHMBase(object):
    pass


def create_shm_array(shm_mem_name, arr):
    # WARNING: MEMORY LEAKAGE
    shm_mem = shared_memory.SharedMemory(
        name=shm_mem_name,
        create=True, size=arr.nbytes)

    shm_array = np.ndarray(
        arr.shape, dtype=arr.dtype, buffer=shm_mem.buf)
    shm_array[:] = arr[:]
    return shm_array, shm_mem


class MessengerRawStringSHM(MessengerRawString):

    def read_sys(self, raw_string):
        splitted = raw_string.split(',')
        vector = np.asarray(splitted, dtype=NATIVE_NP_FLOAT_TYPE)
        # v is 0-th element
        return vector

    def objs_to_mini_batch_single_vector(self, objs):

        local_arr = np.asarray(objs, dtype=NATIVE_NP_FLOAT_TYPE)
        shm_mem_name = "{}_{}".format(self.ident, self.num_batches)

        shm_array, shm_mem = create_shm_array(shm_mem_name, local_arr)

        if self.args.messenger_single_vector and \
                not self.args.messenger_backprop:
            ret = DatumSHM(
                shm_name=shm_mem_name,
                shape=local_arr.shape,
                H_u=1,
                V=None,
                U=None,
                unique_vs=0)

        else:
            ret = DatumSingleVectorSHM(
                shm_name=shm_mem_name,
                shape=local_arr.shape,
                grads=1,
                unique_vs=0)

        shm_mem.close()
        return ret

    def write_sys(self, v, row):
        string = v + "," + ','.join(row)
        print(string)

    def write_sys_minibatch(self, datum):
        arr = datum.unload()
        # arr = np.char.mod('%.4f', arr)
        data_arr = np.char.mod(CHAR_MOD, arr[:, 1:])
        V = arr[:, 0].astype(int).astype(str)
        # destroy
        datum.clean()
        for i, (v, row) in enumerate(zip(V, data_arr)):
            self.write_sys(v, row)

    def objs_to_mini_batch(self, objs):
        # not implemented yet
        raise NotImplementedError


class MessengerRawStringSHMSparseFirstPipe(MessengerRawStringSHM):
    def write_sys(self, row):
        print(row)

    def write_sys_minibatch(self, datum):
        arr = datum.unload()

        for i, row in enumerate(arr):
            v = int(row[0])
            vector = row[1:]
            index = vector.nonzero()[0]
            values = vector[index]
            length = vector.shape[0]
            values = np.char.mod(CHAR_MOD, values)
            values_string = ','.join(values)
            index = index.astype(str)
            index_string = ','.join(index)

            parts = [str(v), str(length), index_string, values_string]
            string = ';'.join(parts)
            if i == arr.shape[0] - 1:
                # destroy
                datum.clean()
            self.write_sys(string)


class MessengerRawStringSHMSparse(MessengerRawStringSHMSparseFirstPipe):
    def read_sys(self, raw_string):
        splitted = raw_string.split(';')
        v = int(splitted[0])
        length = int(splitted[1])
        vector = np.zeros(length + 1, dtype=NATIVE_NP_FLOAT_TYPE)
        if splitted[2] == '':
            pass
        else:
            index = np.asarray(splitted[2].split(','), dtype=int)
            values = np.asarray(splitted[3].split(','),
                                dtype=NATIVE_NP_FLOAT_TYPE)
            vector[index] = values
        vector[0] = v
        return vector


class MessengerByteSHM(MessengerBase):

    def zmq_init(self):
        super(MessengerByteSHM, self).zmq_init()
        name = 'messenger_frontend_{}'.format(self.args.messenger_idx)
        self.messenger_socket = AsyncSocket(
            self.context,
            zmq.REP,
            name,
            'bind',
            "ipc://@{}".format(name))

    def batch_gen(self, mini_batch_size=None):
        self.num_batches = 0
        while True:

            byte_pack = self.messenger_socket.recv_multipart()
            if self.args.sparse:
                if len(byte_pack) < 4:
                    # print("Imcomplete byte_pack: {}".format(byte_pack), file=sys.stderr)
                    break
                sizes, ids, vecs, keys = byte_pack
            else:
                # print("Received | ident: {}, num_batches: {}, len: {}".format(self.ident, self.num_batches, len(byte_pack)), file=sys.stderr)
                if len(byte_pack) < 3:
                    # print("Imcomplete byte_pack: {}".format(byte_pack), file=sys.stderr)
                    break
                sizes, ids, vecs = byte_pack
            # with open("/tmp/sizes_{}_{}.bytes".format(self.ident, self.num_batches), 'wb') as f:
            #     f.write(sizes)
            # with open("/tmp/ids_{}_{}.bytes".format(self.ident, self.num_batches), 'wb') as f:
            #     f.write(ids)
            # with open("/tmp/vecs_{}_{}.bytes".format(self.ident, self.num_batches), 'wb') as f:
            #     f.write(vecs)
            np_sizes = np.frombuffer(
                sizes, dtype=SOCKET_NP_LONG_TYPE).astype(
                    NATIVE_NP_LONG_TYPE)

            np_ids = np.frombuffer(ids, dtype=SOCKET_NP_LONG_TYPE).astype(
                NATIVE_NP_FLOAT_TYPE)
            np_new_ids = np.frombuffer(ids, dtype=SOCKET_NP_LONG_TYPE).astype(
                NATIVE_NP_LONG_TYPE)
            np_vecs = np.frombuffer(
                vecs, dtype=SOCKET_NP_FLOAT_TYPE).astype(NATIVE_NP_FLOAT_TYPE)

            # .astype(NATIVE_NP_FLOAT_TYPE)
            # print(
            #     "Got: {}, {}, {}".format(np_sizes, np_ids, np_vecs),
            #     file=sys.stderr)
            if self.args.sparse:
                np_keys = np.frombuffer(keys, dtype=SOCKET_NP_LONG_TYPE).astype(
                    NATIVE_NP_LONG_TYPE)
                curr = 0
                np_zeros = np.zeros((np_new_ids.shape[0], np_sizes[1]),
                                    dtype=np.float32)
                for j, size in enumerate(np_sizes[2::3]):
                    np_zeros[j, np_keys[curr:curr + size]
                             ] = np_vecs[curr:curr + size]

                    curr += size
                np_vecs = np_zeros

            else:
                np_vecs = np_vecs.reshape((np_new_ids.shape[0], np_sizes[1]))
            np_vecs = np.c_[np_ids, np_vecs].astype(
                    NATIVE_NP_FLOAT_TYPE, copy=False)

            shm_mem_name = "{}_{}_{}".format(
                self.ident, os.getpid(), self.num_batches)
            shm_mem_name_ids = "{}_{}_{}_ids".format(
                self.ident, os.getpid(), self.num_batches)
            # print(
            #     "np_vecs: {}, {}".format(
            #         np_vecs, np_vecs.dtype.byteorder), file=sys.stderr)
            # print(
            #     "np_vecs_bytes: {}".format(np_vecs.tobytes()),
            #     file=sys.stderr)
            shm_array, shm_mem = create_shm_array(shm_mem_name, np_vecs)

            shm_array_ids, shm_mem_ids = create_shm_array(
                shm_mem_name_ids, np_new_ids)
            # print(
            #     "shm_array: {}, {}".format(
            #         shm_array, shm_array.dtype.byteorder), file=sys.stderr)
            # print(
            #     "shm_array_bytes: {}".format(shm_array.tobytes()),
            #     file=sys.stderr)
            if self.args.messenger_single_vector:
                if self.args.messenger_backprop:
                    ret = DatumSingleVectorSHM(
                        shm_name=shm_mem_name,
                        shm_name_ids=shm_mem_name_ids,
                        shape=np_vecs.shape,
                        shape_ids=np_ids.shape,
                        grads=1,
                        unique_vs=0)

                else:
                    ret = DatumSHM(
                        shm_name=shm_mem_name,
                        shm_name_ids=shm_mem_name_ids,
                        shape=np_vecs.shape,
                        shape_ids=np_ids.shape,
                        H_u=1,
                        V=None,
                        U=None,
                        unique_vs=0)

            else:
                ret = DatumSingleVectorSHM(
                    shm_name=shm_mem_name,
                    shm_name_ids=shm_mem_name_ids,
                    shape=np_vecs.shape,
                    shape_ids=np_ids.shape,
                    grads=1,
                    unique_vs=0)

            shm_mem.close()
            shm_mem_ids.close()
            self.num_batches += 0

            # # debug
            # print(
            #     "Immediate unload", file=sys.stderr)
            # ret.unload()
            # print("Packed | ident: {}, num_batches: {}".format(self.ident, self.num_batches), file=sys.stderr)
            yield ret

    def write_sys_minibatch(self, datum):
        arr, arr_ids = datum.unload()
        # arr = np.char.mod('%.4f', arr)
        # data_arr = np.char.mod(CHAR_MOD, arr[:, 1:])
        # print(arr, file=sys.stderr)
        # np_new_ids = arr[:, 0].astype(int)
        np_new_ids = arr_ids
        np_new_vecs = arr[:, 1:]
        if self.args.sparse:
            np_new_sparse_vecs = []
            np_new_sizes = []
            np_new_keys = []

            for i in range(np_new_vecs.shape[0]):
                np_new_sizes.append(1)
                vector = np_new_vecs[i]
                index = vector.nonzero()[0]
                values = vector[index]
                np_new_keys.append(index)
                np_new_sparse_vecs.append(values)
                np_new_sizes.append(np_new_vecs.shape[1])
                np_new_sizes.append(index.shape[0])
                

            np_new_sizes = np.asarray(np_new_sizes)
            np_new_sparse_vecs = np.concatenate(np_new_sparse_vecs)
            np_new_keys = np.concatenate(np_new_keys)
            ret_byte_pack = [
                np_new_sizes.astype(SOCKET_NP_LONG_TYPE),
                np_new_ids.astype(SOCKET_NP_LONG_TYPE),
                np_new_sparse_vecs.astype(SOCKET_NP_FLOAT_TYPE),
                np_new_keys.astype(SOCKET_NP_LONG_TYPE)
            ]

        else:
            np_new_sizes = []
            for i in range(np_new_vecs.shape[0]):
                np_new_sizes.append(1)
                np_new_sizes.append(np_new_vecs.shape[1])

            np_new_sizes = np.asarray(np_new_sizes)

            ret_byte_pack = [
                np_new_sizes.astype(SOCKET_NP_LONG_TYPE),
                np_new_ids.astype(SOCKET_NP_LONG_TYPE),
                np_new_vecs.astype(SOCKET_NP_FLOAT_TYPE)
            ]
            # print(np_new_ids, file=sys.stderr)

        self.messenger_socket.send_multipart(ret_byte_pack)
        # destroy
        datum.clean()
        # print("ReceivedSending | ident: {}, num_batches: {}".format(self.ident, self.num_batches), file=sys.stderr)
        

    def final_write(self):
        return self.write_dummy()

    def write_dummy(self):
        self.messenger_socket.send(b'')
