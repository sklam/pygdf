import os
import multiprocessing


from numba import cuda

from distributed import worker_client, get_client, get_worker
from distributed.client import _get_global_client, _global_client


_USE_IPC = bool(int(os.environ.get('PYGDF_USE_IPC', '0')))



def serialize_gpu_data(gpudata):
    if _USE_IPC:
        return IpcGpuData(gpudata)
    else:
        return gpudata.copy_to_host()


def _get_context():
    pid = multiprocessing.current_process().pid
    ctxid = cuda.current_context().handle.value
    return pid, ctxid


_keepalive = []


def _hash_ipc_handle(ipchandle):
    return hex(hash(tuple(ipchandle._ipc_handle.handle)))


class IpcGpuData(object):
    def __init__(self, gpu_data):
        self._context = _get_context()
        self._gpu_data = gpu_data

    def __reduce__(self):
        args = (self._context, self._gpu_data.get_ipc_handle())
        _keepalive.append((self._gpu_data, args))
        print('serializing', _hash_ipc_handle(args[1]), 'size:', self._gpu_data.size)
        return rebuild_gpu_data, args


_cache = {}


def rebuild_gpu_data(context, ipchandle):
    ####
    try:
        print("worker?", get_worker())
    except ValueError as e:
        print(e)
        print("try client")
        # cl = _global_client[0]
        # print(cl.run(os.getpid))
    else:
        with worker_client() as e:
            print(e.run(os.getpid))

    ####
    if context != _get_context():
        hkey = tuple(ipchandle._ipc_handle.handle)
        if hkey in _cache:
            return _cache[hkey]
        else:
            data = ipchandle.open()
            _keepalive.append(ipchandle)
            _cache[hkey] = data
            print('rebuild', _hash_ipc_handle(ipchandle), 'size:', data.size)
            return data
    else:
        raise NotImplementedError('same context')

