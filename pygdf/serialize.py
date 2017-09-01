import os
import pickle
import multiprocessing
import logging
logging.basicConfig(level=logging.INFO)
logging.getLogger('numba').setLevel(logging.ERROR)
logger = logging.getLogger(__name__)

from numba import cuda

from distributed import worker_client, get_client, get_worker
from distributed.client import _get_global_client, _global_client

import zmq
import socket
import threading


_global_port = [None]
_global_addr = [None]



def init_server():
    _global_addr[0] = socket.gethostname()
    logger.info("host addr: %s", _global_addr[0])

    th = threading.Thread(target=server_loop)
    th.daemon = True
    th.start()


def server_loop():
    logger.info("server loop starts")
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    selected_port = socket.bind_to_random_port("tcp://*")
    _global_port[0] = selected_port
    logger.info('bind to port: %s', selected_port)

    with cuda.gpus[0]:
        while True:
            req = socket.recv()
            out = _handle_request(req)
            socket.send(out)


_USE_IPC = bool(int(os.environ.get('PYGDF_USE_IPC', '1')))


def serialize_gpu_data(gpudata):
    if _USE_IPC:
        return IpcGpuData(gpudata)
    else:
        return gpudata.copy_to_host()


def _get_context():
    pid = multiprocessing.current_process().pid
    ctxid = cuda.current_context().handle.value
    return pid, ctxid


_out_cache = {}


def _hash_ipc_handle(ipchandle):
    return str(hex(hash(tuple(ipchandle._ipc_handle.handle)))).encode()


def _get_key(gpudata):
    return str(hash(gpudata)).encode()


class IpcGpuData(object):
    def __init__(self, gpu_data):
        self._context = _get_context()
        self._gpu_data = gpu_data

    def __reduce__(self):
        remoteinfo = _global_addr[0], _global_port[0]
        # ipch = self._gpu_data.get_ipc_handle()
        key = _get_key(self._gpu_data)
        args = (self._context, key, remoteinfo)
        _out_cache[key] = self._gpu_data
        print('serializing', key, 'size:', self._gpu_data.size)
        return rebuild_gpu_data, args


def _handle_request(req):
    method, key = pickle.loads(req)
    data = _out_cache[key]
    if method == 'NET':
        # NET
        return pickle.dumps(data.copy_to_host())
    else:
        # IPC
        return pickle.dumps(data.get_ipc_handle())


def _request_transfer(key, remoteinfo):
    logger.info("rebuild from: %s for %r", remoteinfo, key)

    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.connect("tcp://{0}:{1}".format(*remoteinfo))

    myaddr = _global_addr[0]
    theiraddr = remoteinfo[0]
    if myaddr == theiraddr:
        # Same machine go by IPC
        logger.info("request by IPC")
        socket.send(pickle.dumps(('IPC', key)))
        rcv = socket.recv()
        ipch = pickle.loads(rcv)
        # Open IPC and copy to local context
        with ipch as data:
            copied = cuda.device_array_like(data)
            copied.copy_to_device(data)
            return copied
    else:
        # Different machine go by NET
        logger.info("request by NET: %s->%s", theiraddr, myaddr)
        socket.send(pickle.dumps(('NET', key)))
        rcv = socket.recv()
        return cuda.to_device(pickle.loads(rcv))


def rebuild_gpu_data(context, key, remoteinfo):
    if context != _get_context():
        out = _request_transfer(key, remoteinfo)
        return out

    else:
        raise NotImplementedError('same process & cuda-context')


init_server()

