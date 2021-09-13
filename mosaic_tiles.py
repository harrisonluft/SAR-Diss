import rasterio
from rasterio.merge import merge
from rasterio.plot import show
import glob
import os

# adapted from https://automating-gis-processes.github.io/CSC18/lessons/L6/raster-mosaic.html

patchPath = r"C:\Users\17075\PycharmDissertation\Images\Patches\pred_mask\georef"
mosaicPath = r"C:\Users\17075\PycharmDissertation\Images\Full Size\mosaics\CZU Ablation\CZU Ablation result.tif"
file_string = "2*.tif"
path = os.path.join(patchPath,file_string)
files = glob.glob(path)

files_mosaic= []
for fps in files:
    src = rasterio.open(fps)
    files_mosaic.append(src)

mosaic, out_trans = merge(files_mosaic)

out_meta = src.meta.copy()
print(out_meta)

out_meta.update({"driver": "GTiff",
                "height": mosaic.shape[1],
                "width": mosaic.shape[2],
                "transform": out_trans,
                "crs": "+proj=utm +zone=10 +ellps=GRS80 +units=m +no_defs "
            })

with rasterio.open(mosaicPath, "w", **out_meta) as dest:
    dest.write(mosaic)