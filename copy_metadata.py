import rasterio
from rasterio.plot import reshape_as_raster
from rasterio.merge import merge
from rasterio.plot import show
import numpy as np
import glob
import os

# adapted from https://gis.stackexchange.com/questions/403276/copy-metadata-from-one-raster-to-another-in-rasterio-using-python

patchPath = r"C:\Users\17075\PycharmDissertation\Images\Patches\test_mask\CZU"
predPath = r"C:\Users\17075\PycharmDissertation\Images\Patches\pred_mask\CZU Ablation"
newMetaPath = r"C:\Users\17075\PycharmDissertation\Images\Patches\pred_mask\georef"
file_string = "2*.tif"
path = os.path.join(patchPath, file_string)
files = glob.glob(path)
names = [os.path.basename(x) for x in files]



for name in names:
    src = rasterio.open(os.path.join(patchPath, name))
    out_meta = src.meta.copy()
    out_transform = src.transform
    out_height = src.height
    out_width = src.width
    crs = src.crs
    out_meta.update({"driver":"GTiff",
                    "height": out_height,
                     "weight": out_width,
                    "transform": out_transform,
                    "crs" : src.crs,
                    "count":1})

    out_tif = os.path.join(predPath, name)

    # read in raster with no metadata
    pred_raster = rasterio.open(out_tif).read()

    # update shape to CHW
    raster = reshape_as_raster(pred_raster)
    raster = np.moveaxis(raster,0,-1)

    # read in path for new file
    new_meta_raster = os.path.join(newMetaPath, name)

    # export raster with updated georeferenced metadata from other file
    with rasterio.open(new_meta_raster, "w", **out_meta) as dest:
        dest.write(raster)
