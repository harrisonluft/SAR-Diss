import os
from os import walk
import numpy as np
from itertools import product
import rasterio as rio
from rasterio import windows
import torch
from skimage import io
from sklearn.feature_extraction import image

# from https://gis.stackexchange.com/questions/285499/how-to-split-multiband-image-into-image-tiles-using-rasterio
folder = 'input_data_dmap/CZU_ablation'
in_path = 'C:/Users/17075/PycharmDissertation/Images/Full Size/'+folder
out_path = 'C:/Users/17075/PycharmDissertation/Images/Patches/'+folder
output_filename = '{}_{}.tif'


def get_tiles(ds, width=128, height=128):
    nols, nrows = ds.meta['width'], ds.meta['height']
    offsets = product(range(0, nols, width), range(0, nrows, height))
    big_window = windows.Window(col_off=0, row_off=0, width=nols, height=nrows)
    for col_off, row_off in  offsets:
        window =windows.Window(col_off=col_off, row_off=row_off, width=width, height=height).intersection(big_window)
        transform = windows.transform(window, ds.transform)
        yield window, transform

dim_pix = 128

# patch training data
filenames = next(walk(in_path), (None, None, []))[2]  # [] if no file

for x in filenames:
    n = 0
    with rio.open(os.path.join(in_path, x)) as inds:
        tile_width, tile_height = 128, 128
        meta = inds.meta.copy()
        for window, transform in get_tiles(inds):
            # print(window)
            meta['transform'] = transform
            meta['width'], meta['height'] = window.width, window.height
            if meta['width'] == 128 and meta['height']==128:
                outpath = os.path.join(out_path,output_filename.format(x.split('.')[0], n))
                with rio.open(outpath, 'w', **meta) as outds:
                    outds.write(inds.read(window=window))
                    n+=1
            else:
                print("too small")

# patch=(128,128)
# filenames = next(walk(in_path), (None, None, []))[2]  # [] if no file
# for x in filenames:
#     img = np.nan_to_num(io.imread(os.path.join(in_path, x)))
#     r = img.unfold(2,h,1),