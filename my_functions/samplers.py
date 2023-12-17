from torchgeo.samplers.utils import get_random_bounding_box, tile_to_chips
from torchgeo.samplers.single import RandomGeoSampler, GridGeoSampler
from torchgeo.datasets import GeoDataset, BoundingBox
from samplers_utils import path_2_tilePolygon, boundingBox_2_Polygon, boundingBox_2_centralPoint
from torchgeo.samplers.constants import Units
from typing import Optional, Union
from collections.abc import Iterator
import torch
import os
import shapely


# Samplers per Base Datasets

class MyRandomGeoSampler(RandomGeoSampler):
    def __init__(
        self,
        dataset: GeoDataset,
        size: Union[tuple[float, float], float],
        length: Optional[int],
        roi: Optional[BoundingBox] = None,
        units: Units = Units.PIXELS,
        verbose: bool = False
    ) -> None:

        super().__init__(dataset, size, length, roi, units)
        self.verbose = verbose
    
    def __iter__(self) -> Iterator[BoundingBox]:
        """Return the index of a dataset.

        Returns:
            (minx, maxx, miny, maxy, mint, maxt) coordinates to index a dataset
        """
        i = 0
        while i < len(self):
            # Choose a random tile, weighted by area
            idx = torch.multinomial(self.areas, 1)
            hit = self.hits[idx]

            tile_path = hit.object
            tile_polyg = path_2_tilePolygon(tile_path)

            bounds = BoundingBox(*hit.bounds) #TODO: ridurre i bounds usando il bbox del geojson
            # Choose a random index within that tile
            bounding_box = get_random_bounding_box(bounds, self.size, self.res)
            #rnd_bbox_polyg = boundingBox_2_Polygon(bounding_box)
            rnd_central_point = boundingBox_2_centralPoint(bounding_box)

            #se il punto centrale della rnd_bbox è nel poligono (definito con geojson) del tile
            if rnd_central_point.intersects(tile_polyg):
                if self.verbose: #TODO: magari in futuro togliere il verbose per velocizzare
                    print('In sampler')
                    print('tile_polyg', tile_polyg)
                    print()
                i += 1
                yield bounding_box
            
            else:
                continue

class MyGridGeoSampler(GridGeoSampler):
    def __init__(self,
        dataset: GeoDataset,
        size: Union[tuple[float, float], float],
        stride: Union[tuple[float, float], float],
        roi: Optional[BoundingBox] = None,
        units: Units = Units.PIXELS,
        ) -> None:

        super().__init__(dataset, size, stride, roi, units)
    
    def __iter__(self) -> Iterator[BoundingBox]:
        """Return the index of a dataset.

        Returns:
            (minx, maxx, miny, maxy, mint, maxt) coordinates to index a dataset
        """
        # For each tile...
        for hit in self.hits: #These hits are all the tiles that intersect the roi (region of interest). If roi not specified then hits = all the tiles
            tile_path = hit.object
            tile_polygon = path_2_tilePolygon(tile_path)

            print('In sampler') #TODO: togliere le print quando usato davvero
            print('tile_polygon: ', tile_polygon)

            bounds = BoundingBox(*hit.bounds)
            rows, cols = tile_to_chips(bounds, self.size, self.stride)
            mint = bounds.mint
            maxt = bounds.maxt

            # For each row...
            for i in range(rows):
                miny = bounds.miny + i * self.stride[0]
                maxy = miny + self.size[0]

                # For each column...
                for j in range(cols):
                    minx = bounds.minx + j * self.stride[1]
                    maxx = minx + self.size[1]
                    selected_bbox = BoundingBox(minx, maxx, miny, maxy, mint, maxt)
                    selected_bbox_polygon = boundingBox_2_Polygon(selected_bbox)
                    if selected_bbox_polygon.intersects(tile_polygon):
                        #print("selected_bbox_polygon", selected_bbox_polygon)
                        yield selected_bbox
                    else:
                        continue

# Samplers per Intersection Datasets
class MyIntersectionRandomGeoSampler(RandomGeoSampler):
    def __init__(
        self,
        dataset: GeoDataset,
        size: Union[tuple[float, float], float],
        length: Optional[int],
        roi: Optional[BoundingBox] = None,
        units: Units = Units.PIXELS,
    ) -> None:

        super().__init__(dataset, size, length, roi, units)
    
    def __iter__(self) -> Iterator[BoundingBox]:
        """Return the index of a dataset.

        Returns:
            (minx, maxx, miny, maxy, mint, maxt) coordinates to index a dataset
        """
        i = 0
        while i < len(self):
            # Choose a random tile, weighted by area
            idx = torch.multinomial(self.areas, 1)
            hit = self.hits[idx]

            tile_path1= hit.object[0]
            tile_path2= hit.object[1]

            tile_polyg1 = path_2_tilePolygon(tile_path1)
            tile_polyg2 = path_2_tilePolygon(tile_path2)

            bounds = BoundingBox(*hit.bounds) #TODO: ridurre i bounds usando il bbox del geojson
            # Choose a random index within that tile
            bounding_box = get_random_bounding_box(bounds, self.size, self.res)
            rnd_bbox_polyg = boundingBox_2_Polygon(bounding_box)                
            rnd_central_point = boundingBox_2_centralPoint(bounding_box)

            #se il centro della bounding_box ricade nel polygono del tile1 e in quello del tile2
            # (calcolati usando il geojson) allora la bounding_box è valida
            if rnd_central_point.intersects(tile_polyg1) and rnd_central_point.intersects(tile_polyg2):
                print('In sampler')
                print('tile_polyg1', tile_polyg1)
                print('tile_polyg2', tile_polyg2)
                print()
                i += 1
                yield bounding_box
            
            else:
                continue


class MyIntersectionGridGeoSampler(GridGeoSampler):
    def __init__(
        self,
        dataset: GeoDataset,
        size: Union[tuple[float, float], float],
        stride: Union[tuple[float, float], float],
        roi: Optional[BoundingBox] = None,
        units: Units = Units.PIXELS,
        ) -> None:

        super().__init__(dataset, size, stride, roi, units)


    def __iter__(self) -> Iterator[BoundingBox]:
        """Return the index of a dataset.

        Returns:
            (minx, maxx, miny, maxy, mint, maxt) coordinates to index a dataset
        """
        # For each tile...
        for hit in self.hits:
            path_tile_1 = hit.object[0]
            path_tile_2 = hit.object[1]
            polyg_tile_1 = path_2_tilePolygon(path_tile_1)
            polyg_tile_2=  path_2_tilePolygon(path_tile_2)

            print('In sampler')
            print('tile_polygon 1: ', polyg_tile_1)
            print('tile_polygon 2: ', polyg_tile_2)

            bounds = BoundingBox(*hit.bounds)
            rows, cols = tile_to_chips(bounds, self.size, self.stride)
            mint = bounds.mint
            maxt = bounds.maxt

            # For each row...
            for i in range(rows):
                miny = bounds.miny + i * self.stride[0]
                maxy = miny + self.size[0]

                # For each column...
                for j in range(cols):
                    minx = bounds.minx + j * self.stride[1]
                    maxx = minx + self.size[1]
                    selected_bbox = BoundingBox(minx, maxx, miny, maxy, mint, maxt)
                    selected_bbox_polygon = boundingBox_2_Polygon(selected_bbox)
                    if selected_bbox_polygon.intersects(polyg_tile_1) and selected_bbox_polygon.intersects(polyg_tile_2):
                        print("selected_bbox_polygon", selected_bbox_polygon)
                        yield selected_bbox
                    else:
                        continue