# coding: utf-8
import os
os.environ['PYGDF_USE_IPC'] = '1'
import dask_gdf as dgd
from dask.distributed import Client
import dask
import pygdf
import numpy as np

def main():
    df = pygdf.DataFrame()
    nelem = 1000 #20 * 10**4
    df['a'] = np.arange(nelem)
    df['b'] = np.arange(nelem)

    print(df)

    client = Client('0.0.0.0:8786')
    with client:
        dask.set_options(get=client.get)

        print('here')
        gd = dgd.from_pygdf(df, npartitions=2)
        q1 = gd.query('a > 2')
        out = q1.compute().to_pandas()

        print(out.head())



if __name__ == '__main__':
    main()

# from numba import cuda
# def close_profile():
#     cuda.profile_stop()
# client.run(close_profile)
# client.close()

