# coding: utf-8
import os
# os.environ['PYGDF_USE_IPC'] = '1'
from numba import cuda
cuda.select_device(0)
import dask_gdf as dgd
from dask.distributed import Client
import dask
import pygdf
import numpy as np


def main():

    client = Client('nvidia1:8786')

    with client:
        dask.set_options(get=client.get)

        df = pygdf.DataFrame()
        nelem = 1 * 10**6
        df['a'] = np.arange(nelem)
        df['b'] = np.arange(nelem)

        print(df.head())

        print('here')
        gd = dgd.from_pygdf(df, npartitions=10)
        q1 = gd.query('a > 2')
        c = q1.a + q1.b

        print(c.mean().compute())


if __name__ == '__main__':
    main()

# from numba import cuda
# def close_profile():
#     cuda.profile_stop()
# client.run(close_profile)
# client.close()

