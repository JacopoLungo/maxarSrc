import glob
from pathlib import Path
import sys
import pyproj
import geopandas as gpd
from pyquadkey2 import quadkey
import pandas as pd
from shapely import geometry


def get_mosaics_names(event_name, data_root = '/mnt/data2/vaschetti_data/maxar', when = None):
    """
    Get all the mosaic names for an event.
    Input:
        event_name: Example: 'Gambia-flooding-8-11-2022'
        data_root: Example: '/mnt/data2/vaschetti_data/maxar'
        when: 'pre' or 'post'. Default matches both
    Output:
        all_mosaics: List of mosaic names. Example: ['104001007A565700', '104001007A565800']
    """
    data_root = Path(data_root)
    all_mosaics = []
    if when is not None:
        for mosaic_name in glob.glob('*', root_dir=data_root/event_name/when):
            all_mosaics.append(mosaic_name)
    else:
        for mosaic_name in glob.glob('**/*', root_dir=data_root/event_name):
            all_mosaics.append(mosaic_name.split('/')[1])
    return all_mosaics

def get_mosaic_bbox(event_name, mosaic_name, path_mosaic_metatada = '/home/vaschetti/maxarSrc/metadata/from_github_maxar_metadata/datasets', return_proj_coords = False):
    """
    Get the bbox of a mosaic. It return the coordinates of the bottom left and top right corners.
    Input:
        event_name: Example: 'Gambia-flooding-8-11-2022'
        mosaic_name: It could be an element of the output of get_mosaics_names(). Example: '104001007A565700'
        path_mosaic_metatada: Path to the folder containing the geojson
        return_proj_coords: If True, it returns the coordinates in the projection of the mosaic.
    """
    
    path_mosaic_metatada = Path(path_mosaic_metatada)
    file_name = mosaic_name + '.geojson'
    geojson_path = path_mosaic_metatada / event_name / file_name
    gdf = gpd.read_file(geojson_path)

    minx = sys.maxsize
    miny = sys.maxsize
    maxx = 0
    maxy = 0

    for _, row in gdf.iterrows():
        tmp_minx, tmp_miny, tmp_maxx, tmp_maxy = [float(el) for el in row['proj:bbox'].split(',')]
        if tmp_minx < minx:
            minx = tmp_minx
        if tmp_miny < miny:
            miny = tmp_miny
        if tmp_maxx > maxx:
            maxx = tmp_maxx
        if tmp_maxy > maxy:
            maxy = tmp_maxy
    
    if not return_proj_coords:
        source_crs = gdf['proj:epsg'].values[0]
        target_crs = pyproj.CRS('EPSG:4326')
        transformer = pyproj.Transformer.from_crs(source_crs, target_crs)

        bott_left_lat, bott_left_lon = transformer.transform(minx, miny)
        top_right_lat, top_right_lon = transformer.transform(maxx, maxy)
        
        return (bott_left_lon, bott_left_lat), (top_right_lon, top_right_lat)
    
    return (minx, miny), (maxx, maxy)

def intersecting_qks(bott_left_lon_lat: tuple, top_right_lon_lat: tuple, min_level=7, max_level=9):
    """
    Get the quadkeys that intersect the given bbox.
    Input:
        bott_left_lon_lat: Tuple with the coordinates of the bottom left corner. Example: (-16.5, 13.5)
        top_right_lon_lat: Tuple with the coordinates of the top right corner. Example: (-15.5, 14.5)
        min_level: Minimum level of the quadkeys
        max_level: Maximum level of the quadkeys
    """
    bott_left_lon, bott_left_lat = bott_left_lon_lat
    top_right_lon, top_right_lat = top_right_lon_lat
    qk_bott_left = quadkey.from_geo((bott_left_lat, bott_left_lon), level=max_level) #lat, lon
    qk_top_right = quadkey.from_geo((top_right_lat, top_right_lon), level=max_level) #lat, lon
    hits = qk_bott_left.difference(qk_top_right)
    candidate_hits = set()

    for hit in hits:
        current_qk = hit
        for _ in range(min_level, max_level):
            current_qk = current_qk.parent()
            candidate_hits.add(current_qk)
    hits.extend(candidate_hits)
    return [int(str(hit)) for hit in hits]

def qk_building_gdf(qk_list, csv_path = 'metadata/buildings_dataset_links.csv', dataset_crs = None, quiet = False):
    """
    Returns a geodataframe with the buildings of the country passed as input.
    It downloads the dataset from a link in the dataset-links.csv file.
    Coordinates are converted in the crs passed as input.
    Inputs:
        qk_list: the list of quadkeys to look for in the csv
        root: the root directory of the dataset-links.csv file
        dataset_crs: the crs in which to convert the coordinates of the buildings
        quiet: if True, it doesn't print anything
    """
    building_links_df = pd.read_csv(csv_path)
    country_links = building_links_df[building_links_df['QuadKey'].isin(qk_list)]

    if not quiet:
        print(f"Found {len(country_links)} links matching: {qk_list}")

    gdfs = []
    for _, row in country_links.iterrows():
        df = pd.read_json(row.Url, lines=True)
        df["geometry"] = df["geometry"].apply(geometry.shape)
        gdf_down = gpd.GeoDataFrame(df, crs=4326)
        gdfs.append(gdf_down)

    gdfs = pd.concat(gdfs)
    if dataset_crs is not None: #se inserito il crs del dataset, lo converto
        gdfs = gdfs.to_crs(dataset_crs)
    return gdfs
