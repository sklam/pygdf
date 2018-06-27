from __future__ import print_function, division

import numpy as np
from numba import cuda

from . import cudautils, utils
from .serialize import register_distributed_serializer


class Buffer(object):
    """A 1D gpu buffer.
    """
    _cached_ipch = None

    @classmethod
    def from_empty(cls, mem):
        """From empty device array
        """
        return cls(mem, size=0, capacity=mem.size)

    @classmethod
    def null(cls, dtype):
        """Create a "null" buffer with a zero-sized device array.
        """
        mem = cuda.device_array(0, dtype=dtype)
        return cls(mem, size=0, capacity=0)

    def __init__(self, mem, size=None, capacity=None):
        if size is None:
            size = mem.size
        if capacity is None:
            capacity = size
        self.mem = cudautils.to_device(mem)
        _BufferSentry(self.mem).ndim(1)
        self.size = size
        self.capacity = capacity
        self.dtype = self.mem.dtype

    def serialize(self, serialize, context=None):
        from .serialize import should_use_ipc

        use_ipc = should_use_ipc(context)
        print('use_ipc', use_ipc)
        header = {}
        if use_ipc:
            if self._cached_ipch is not None:
                ipch = self._cached_ipch
            else:
                ipch = self.to_gpu_array().get_ipc_handle()
            header['kind'] = 'ipc'
            header['mem'], frames = serialize(ipch)
            # Keep IPC handle alive
            self._cached_ipch = ipch
        else:
            header['kind'] = 'normal'
            header['mem'], frames = serialize(self.to_array())
        return header, frames

    @classmethod
    def deserialize(cls, deserialize, header, frames):
        if header['kind'] == 'ipc':
            ipch = deserialize(header['mem'], frames)
            with ipch as data:
                mem = cuda.device_array_like(data)
                mem.copy_to_device(data)
        else:
            mem = deserialize(header['mem'], frames)
            mem.flags['WRITEABLE'] = True  # XXX: hack for numba to work
        return Buffer(mem)

    def __reduce__(self):
        cpumem = self.to_array()
        # Note: pickled Buffer only stores *size* element.
        return type(self), (cpumem,)

    def __sizeof__(self):
        return int(self.mem.alloc_size)

    def __getitem__(self, arg):
        if isinstance(arg, slice):
            sliced = self.to_gpu_array()[arg]
            return Buffer(sliced)
        elif isinstance(arg, int):
            arg = utils.normalize_index(arg, self.size)
            return self.mem[arg]
        else:
            raise NotImplementedError(type(arg))

    @property
    def avail_space(self):
        return self.capacity - self.size

    def _sentry_capacity(self, size_needed):
        if size_needed > self.avail_space:
            raise MemoryError('insufficient space in buffer')

    def append(self, element):
        self._sentry_capacity(1)
        self.extend(np.asarray(element, dtype=self.dtype))

    def extend(self, array):
        needed = array.size
        self._sentry_capacity(needed)
        array = cudautils.astype(array, dtype=self.dtype)
        self.mem[self.size:].copy_to_device(array)
        self.size += needed

    def astype(self, dtype):
        if self.dtype == dtype:
            return self
        else:
            return Buffer(cudautils.astype(self.mem, dtype=dtype))

    def to_array(self):
        return self.to_gpu_array().copy_to_host()

    def to_gpu_array(self):
        return self.mem[:self.size]

    def copy(self):
        """Deep copy the buffer
        """
        return Buffer(mem=cudautils.copy_array(self.mem),
                      size=self.size, capacity=self.capacity)

    def as_contiguous(self):
        out = Buffer(mem=cudautils.as_contiguous(self.mem),
                     size=self.size, capacity=self.capacity)
        assert out.is_contiguous()
        return out

    def is_contiguous(self):
        return self.mem.is_c_contiguous()


class BufferSentryError(ValueError):
    pass


class _BufferSentry(object):
    def __init__(self, buf):
        self._buf = buf

    def dtype(self, dtype):
        if self._buf.dtype != dtype:
            raise BufferSentryError('dtype mismatch')
        return self

    def ndim(self, ndim):
        if self._buf.ndim != ndim:
            raise BufferSentryError('ndim mismatch')
        return self

    def contig(self):
        if not self._buf.is_c_contiguous():
            raise BufferSentryError('non contiguous')


register_distributed_serializer(Buffer)
