# ==========================================
#                DESCRIPTION
# ==========================================
# 
# This script allows extracting, georeferencing and merging tile-based Web Mercator image maps based on user-defined boundaries.
# It is possible to address entirely open web servers, or providers requring authentifiaction through logins or API keys.
# Data access and extraction should always happen in accordance with the provider's Terms of Use as wll as local laws and regulations.
#
# ==========================================

import os
import math
import requests
import numpy as np
import geopandas as gpd
import rasterio
from io import BytesIO
from PIL import Image
from pyproj import Transformer
from rasterio.transform import from_origin
from rasterio.crs import CRS

# ==========================================
#                USER CONFIG
# ==========================================
#
# URL Settings
BASE_URL_TEMPLATE = "{BASE_URL}/{zoom_level}/{tile_x}/{tile_y}" #replace {BASE_URL} with the base URL of the desired map provider
FILE_EXTENSION    = "png"  # the image format of the desired map as given in the end of the provider's URL, e.g., png, jpg, webp
API_QUERY     = ""  # possible API key at the end of the URL; leave as "" if no API key is required

# Input/Output Paths
BOUNDARY_PATH = "" # gpkg file with the desired boundaries for extraction
COOKIE_FILE   = "cookies.txt"  # cookie file in Netscape format, containing e.g. session-sepcific authentification credentials. Set to None if not using cookies
OUTPUT_TIFF   = "output.tif" # output raster
TEMP_DIR      = "temp_tiles" # folder for temporary saving extracted image tiles

# Map Settings
ZOOM_LEVEL = 8 # zoom level
# ==========================================


def lonlat_to_webmercator(lon, lat):
    """Converts longitude and latitude to Web Mercator (EPSG:3857)."""
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
    return transformer.transform(lon, lat)

def webmercator_to_tile_google(x, y, zoom):
    """Converts Web Mercator coordinates to tiling (x, y) indices."""
    tile_size = 256
    origin_shift = 2 * math.pi * 6378137 / 2.0
    resolution = (2 * math.pi * 6378137) / tile_size / (2**zoom)
    tile_x = int((x + origin_shift) / resolution / tile_size)
    tile_y = int((origin_shift - y) / resolution / tile_size)
    return tile_x, tile_y

def tile_to_web_mercator_bounds(zoom, x_tile, y_tile):
    """Calculates the bounding box in Web Mercator for a specific tile."""
    tile_size = 256
    initial_resolution = 2 * np.pi * 6378137 / tile_size
    origin_shift = 2 * np.pi * 6378137 / 2.0
    resolution = initial_resolution / (2**zoom)
    
    min_x = x_tile * tile_size * resolution - origin_shift
    max_y = origin_shift - y_tile * tile_size * resolution
    max_x = (x_tile + 1) * tile_size * resolution - origin_shift
    min_y = origin_shift - (y_tile + 1) * tile_size * resolution
    return min_x, min_y, max_x, max_y

def load_cookies_from_file(file_path):
    """Loads cookies from a Netscape cookie file."""
    cookies = {}
    with open(file_path, 'r') as file:
        for line in file:
            if not line.startswith('#') and line.strip():
                parts = line.strip().split('\t')
                if len(parts) >= 7:
                    domain, flag, path, secure, expiration, name, value = parts
                    cookies[name] = value
    return cookies

def merge_tiffs(tiff_files, output_path):
    """Merges a list of temporary GeoTIFFs into one large mosaic."""
    srcs = [rasterio.open(tiff) for tiff in tiff_files]
    meta = srcs[0].meta.copy()

    left = min([src.bounds.left for src in srcs])
    bottom = min([src.bounds.bottom for src in srcs])
    right = max([src.bounds.right for src in srcs])
    top = max([src.bounds.top for src in srcs])

    dst_width = math.ceil((right - left) / srcs[0].res[0])
    dst_height = math.ceil((top - bottom) / srcs[0].res[1])
    
    meta.update({
        'width': dst_width,
        'height': dst_height,
        'transform': from_origin(left, top, srcs[0].res[0], srcs[0].res[1]),
    })

    with rasterio.open(output_path, 'w', **meta) as dst:
        for src in srcs:
            x_off = int((src.bounds.left - left) / src.res[0])
            y_off = int((top - src.bounds.top) / src.res[1])
            dst.write(src.read(), window=((y_off, y_off + src.height), (x_off, x_off + src.width)))
    
    for src in srcs:
        src.close()

def main():
    # 1. Determine Tile Range from Boundary
    gdf = gpd.read_file(BOUNDARY_PATH)
    bounds = gdf.total_bounds 
    
    min_x_wm, min_y_wm = lonlat_to_webmercator(bounds[0], bounds[1])
    max_x_wm, max_y_wm = lonlat_to_webmercator(bounds[2], bounds[3])
    
    start_x, start_y = webmercator_to_tile_google(min_x_wm, max_y_wm, ZOOM_LEVEL)
    end_x, end_y = webmercator_to_tile_google(max_x_wm, min_y_wm, ZOOM_LEVEL)

    # 2. Setup Session
    session = requests.Session()
    if COOKIE_FILE:
        session.cookies.update(load_cookies_from_file(COOKIE_FILE))

    if not os.path.exists(TEMP_DIR):
        os.makedirs(TEMP_DIR)

    # 3. Download and Georeference Tiles
    tiff_files = []
    tile_coords = [(x, y) for x in range(start_x, end_x + 1) for y in range(start_y, end_y + 1)]
    
    print(f"Total tiles to process: {len(tile_coords)}")

    for i, (tx, ty) in enumerate(tile_coords):
        # Format the URL using the template and config variables
        url = f"{BASE_URL_TEMPLATE.format(zoom_level=ZOOM_LEVEL, tile_x=tx, tile_y=ty)}.{FILE_EXTENSION}?{API_QUERY}"
        
        try:
            resp = session.get(url)
            resp.raise_for_status()

            img = Image.open(BytesIO(resp.content))
            data = np.array(img)
            
            min_x, min_y, max_x, max_y = tile_to_web_mercator_bounds(ZOOM_LEVEL, tx, ty)
            transform = from_origin(min_x, max_y, (max_x - min_x) / img.width, (max_y - min_y) / img.height)
            
            temp_path = os.path.join(TEMP_DIR, f"tile_{tx}_{ty}.tif")
            
            with rasterio.open(
                temp_path, 'w', driver='GTiff',
                height=data.shape[0], width=data.shape[1],
                count=data.shape[2] if data.ndim == 3 else 1,
                dtype=data.dtype, crs=CRS.from_epsg(3857), transform=transform
            ) as dst:
                if data.ndim == 3:
                    for band in range(data.shape[2]):
                        dst.write(data[:, :, band], band + 1)
                else:
                    dst.write(data, 1)
            
            tiff_files.append(temp_path)
            print(f"Processed {i+1}/{len(tile_coords)}: Tile {tx}, {ty}")
            
        except Exception as e:
            print(f"Skipping tile {tx}, {ty} due to error: {e}")

    # 4. Merge and Cleanup
    if tiff_files:
        print("Merging tiles...")
        merge_tiffs(tiff_files, OUTPUT_TIFF)
        for f in tiff_files:
            os.remove(f)
        try:
            os.rmdir(TEMP_DIR)
        except OSError:
            pass
        print(f"Success! Output saved to {OUTPUT_TIFF}")
    else:
        print("No tiles were successfully processed.")

if __name__ == "__main__":
    main()