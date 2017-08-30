import os
import multiprocessing

from numba import cuda


_USE_IPC = bool(int(os.environ.get('PYGDF_USE_IPC', '0')))


def serialize_gpu_data(gpudata):
    if _USE_IPC:
        return IpcGpuData(gpudata)
    else:
        return gpudata.copy_to_host()


def _get_context():
    pid = multiprocessing.current_process().pid
    ctxid = cuda.current_context().handle.value
    return pid, ctxid, str(cuda.get_current_device())


_keepalive = []


class IpcGpuData(object):
    def __init__(self, gpu_data):
        self._context = _get_context()
        self._gpu_data = gpu_data

    def __reduce__(self):
        args = (self._context, self._gpu_data.get_ipc_handle())
        ipchandle = args[1]
        _keepalive.append((self._gpu_data, args), 'hash', hash(tuple(ipchandle._ipc_handle.handle)))
        print("serializing", self._context)
        return rebuild_gpu_data, args


_cache = {}


def rebuild_gpu_data(context, ipchandle):
    if context != _get_context():
        hkey = tuple(ipchandle._ipc_handle.handle)
        if hkey in _cache:
            return _cache[hkey]
        else:
            hashd = hash(tuple(ipchandle._ipc_handle.handle))
            print('rebuild', hashd, 'from', context, 'here', _get_context())
            data = ipchandle.open()
            _keepalive.append(ipchandle)
            _cache[hkey] = data
            return data
    else:
        raise NotImplementedError('same context')

