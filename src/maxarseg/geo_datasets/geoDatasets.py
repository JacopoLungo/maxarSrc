from torchgeo.datasets import RasterDataset, IntersectionDataset, BoundingBox
import torch
import matplotlib.pyplot as plt
import os
from typing import Any, cast
from torch import Tensor
import re
import sys
import rasterio as rio
from maxarseg.samplers import samplers_utils
from rasterio.features import rasterize


#TODO: pulire il codice sotto dai commenti
class MxrSingleTile(RasterDataset):
    """
    A dataset for reading a single tile.
    Returns a dict with:
        - crs
        - bbox of the sampled patch
        - image patch
    """
    filename_glob = "*.tif"
    is_image = True
    parent_tile_bbox_in_item = False #this is a parameter to chose if we want to include the parent tile bbox in the return of __getitem__

    def __getitem__(self, query: BoundingBox) -> dict[str, Any]:
        """Retrieve image/mask and metadata indexed by query.

        Args:
            query: (minx, maxx, miny, maxy, mint, maxt) coordinates to index

        Returns:
            sample of image/mask and metadata at that index

        Raises:
            IndexError: if query is not found in the index
        """
        hits = self.index.intersection(tuple(query), objects=True)
        filepaths = cast(list[str], [hit.object for hit in hits])

        if not filepaths:
            raise IndexError(
                f"query: {query} not found in index with bounds: {self.bounds}"
            )

        data = self._merge_files(filepaths, query, self.band_indexes)

        sample = {"crs": self.crs, "bbox": query}

        data = data.to(self.dtype)

        if self.is_image:
            sample["image"] = data
        else:
            sample["mask"] = data

        if self.transforms is not None:
            sample = self.transforms(sample)

        if self.parent_tile_bbox_in_item:
            #initialize bbox
            minx = sys.float_info.max
            maxx = -sys.float_info.max
            miny = sys.float_info.max
            maxy = -sys.float_info.max
            mint = sys.float_info.max
            maxt = -sys.float_info.max

            #if there are multiple hits (tiles) take the largest bbox (include both tiles)
            for hit in hits:
                minx = min(minx, hit[0].minx)
                maxx = max(maxx, hit[1].maxx)
                miny = min(miny, hit[2].miny)
                maxy = max(maxy, hit[3].maxy)
                mint = min(mint, hit[4].mint)
                maxt = max(maxt, hit[5].maxt)


            sample['parent_tile_bbox'] = BoundingBox(minx, maxx, miny, maxy, mint, maxt)


        return sample

    def set_parent_tile_bbox_in_item(self):
        self.parent_tile_bbox_in_item = True


    #tr = Transformer.from_crs("EPSG:32628", "EPSG:4326")
    def plot(self, sample):
        # Find the correct band index order
        #rgb_indices = []
        #for band in self.rgb_bands:
        #    rgb_indices.append(self.all_bands.index(band))

        # Reorder and rescale the image
        #tr = Transformer.from_crs("EPSG:32628", "EPSG:4326")
        minx, maxx, miny, maxy = sample["bbox"].minx, sample["bbox"].maxx, sample["bbox"].miny, sample["bbox"].maxy
        #sx_low =  tr.transform(minx, miny)
        #dx_high = tr.transform(maxx, maxy)
        print('In plot')
        print('Crs', self.crs)
        print('sx_low: ', (minx, miny))
        print('dx_high: ', (maxx, maxy))
        image = sample["image"].permute(1, 2, 0).numpy().astype('uint8')

        # Plot the image
        fig, ax = plt.subplots()
        ax.imshow(image)

        return fig, ax
    
class MxrSingleTileNoEmpty(RasterDataset):
    """
    A dataset for reading a single tile.
    Returns a dict with:
        - crs
        - bbox of the sampled patch
        - offset (index of the top left corner of the patch in the original image)
        - image patch
    """
    
    filename_glob = "*.tif"
    is_image = True
    def __init__(self, paths, tile_aoi_gdf):
        super().__init__(paths)
        with rio.open(self.files[0]) as src:
            self.to_index = src.index
            self.to_xy = src.xy
            self.transform = src.transform
            self.tile_shape = (src.height, src.width)
        
        
        self.tile_aoi_gdf = tile_aoi_gdf
        #here tile aoi must be in proj crs
        self.aoi_mask = rasterize(self.tile_aoi_gdf.geometry, out_shape = self.tile_shape, fill=False, default_value=True, transform = self.transform)
    
    def __getitem__(self, query: BoundingBox) -> dict[str, Any]:
        """Retrieve image/mask and metadata indexed by query.

        Args:
            query: (minx, maxx, miny, maxy, mint, maxt) coordinates to index

        Returns:
            sample of image/mask and metadata at that index

        Raises:
            IndexError: if query is not found in the index
        """
        
        hits = self.index.intersection(tuple(query), objects=True)
        filepaths = cast(list[str], [hit.object for hit in hits])

        if not filepaths:
            raise IndexError(
                f"query: {query} not found in index with bounds: {self.bounds}"
            )

        data = self._merge_files(filepaths, query, self.band_indexes)

        sample = {"crs": self.crs, "bbox": query, "top_lft_index": self.to_index(query[0], query[3])}

        data = data.to(self.dtype)

        if self.is_image:
            sample["image"] = data
        else:
            sample["mask"] = data

        return sample

    #tr = Transformer.from_crs("EPSG:32628", "EPSG:4326")
    def plot(self, sample):
        # Find the correct band index order
        #rgb_indices = []
        #for band in self.rgb_bands:
        #    rgb_indices.append(self.all_bands.index(band))

        # Reorder and rescale the image
        #tr = Transformer.from_crs("EPSG:32628", "EPSG:4326")
        minx, maxx, miny, maxy = sample["bbox"].minx, sample["bbox"].maxx, sample["bbox"].miny, sample["bbox"].maxy
        #sx_low =  tr.transform(minx, miny)
        #dx_high = tr.transform(maxx, maxy)
        print('In plot')
        print('Crs', self.crs)
        print('sx_low: ', (minx, miny))
        print('dx_high: ', (maxx, maxy))
        image = sample["image"].permute(1, 2, 0).numpy().astype('uint8')

        # Plot the image
        fig, ax = plt.subplots()
        ax.imshow(image)

        return fig, ax
    
class Maxar(RasterDataset):
    filename_glob = "*.tif"
    is_image = True
    parent_tile_bbox_in_item = False

    def __getitem__(self, query: BoundingBox) -> dict[str, Any]:
        """Retrieve image/mask and metadata indexed by query.

        Args:
            query: (minx, maxx, miny, maxy, mint, maxt) coordinates to index

        Returns:
            sample of image/mask and metadata at that index

        Raises:
            IndexError: if query is not found in the index
        """
        hits = self.index.intersection(tuple(query), objects=True)
        filepaths = cast(list[str], [hit.object for hit in hits])

        if not filepaths:
            raise IndexError(
                f"query: {query} not found in index with bounds: {self.bounds}"
            )

        data = self._merge_files(filepaths, query, self.band_indexes)

        sample = {"crs": self.crs, "bbox": query}

        data = data.to(self.dtype)

        if self.is_image:
            sample["image"] = data
        else:
            sample["mask"] = data

        if self.transforms is not None:
            sample = self.transforms(sample)

        if self.parent_tile_bbox_in_item:
            #initialize bbox
            minx = sys.float_info.max
            maxx = -sys.float_info.max
            miny = sys.float_info.max
            maxy = -sys.float_info.max
            mint = sys.float_info.max
            maxt = -sys.float_info.max

            #if there are multiple hits (tiles) take the largest bbox (include both tiles)
            for hit in hits:
                minx = min(minx, hit[0].minx)
                maxx = max(maxx, hit[1].maxx)
                miny = min(miny, hit[2].miny)
                maxy = max(maxy, hit[3].maxy)
                mint = min(mint, hit[4].mint)
                maxt = max(maxt, hit[5].maxt)


            sample['parent_tile_bbox'] = BoundingBox(minx, maxx, miny, maxy, mint, maxt)


        return sample

    def set_parent_tile_bbox_in_item(self):
        self.parent_tile_bbox_in_item = True


    #tr = Transformer.from_crs("EPSG:32628", "EPSG:4326")
    def plot(self, sample):
        # Find the correct band index order
        #rgb_indices = []
        #for band in self.rgb_bands:
        #    rgb_indices.append(self.all_bands.index(band))

        # Reorder and rescale the image
        #tr = Transformer.from_crs("EPSG:32628", "EPSG:4326")
        minx, maxx, miny, maxy = sample["bbox"].minx, sample["bbox"].maxx, sample["bbox"].miny, sample["bbox"].maxy
        #sx_low =  tr.transform(minx, miny)
        #dx_high = tr.transform(maxx, maxy)
        print('In plot')
        print('Crs', self.crs)
        print('sx_low: ', (minx, miny))
        print('dx_high: ', (maxx, maxy))
        image = sample["image"].permute(1, 2, 0).numpy().astype('uint8')

        #Da eliminare
        #image = sample["image"].permute(1, 2, 0)
        #image = torch.clamp(image / 300, min=0, max=1).numpy()

        # Plot the image
        fig, ax = plt.subplots()
        ax.imshow(image)

        return fig, ax

class MaxarIntersectionDataset(IntersectionDataset):
    def __init__(self,dataset1, dataset2):
        super().__init__(dataset1, dataset2)

    def _merge_dataset_indices(self) -> None:
        """Create a new R-tree out of the individual indices from two datasets."""
        i = 0
        ds1, ds2 = self.datasets
        for hit1 in ds1.index.intersection(ds1.index.bounds, objects=True):
            for hit2 in ds2.index.intersection(hit1.bounds, objects=True):
                print('In merge')
                print('hit1: ', hit1.object)
                print('hit2: ', hit2.object)
                box1 = BoundingBox(*hit1.bounds)
                box2 = BoundingBox(*hit2.bounds)
                self.index.insert(id = i, coordinates = tuple(box1 & box2), obj = (hit1.object, hit2.object))
                i += 1

        if i == 0:
            raise RuntimeError("Datasets have no spatiotemporal intersection")
    
    def plot(self, sample):
        imgPre = sample["image"][:3].permute(1, 2, 0)
        imgPre = torch.clamp(imgPre / 300, min=0, max=1).numpy()
        imgPost = sample["image"][3:].permute(1, 2, 0)
        imgPost = torch.clamp(imgPost / 300, min=0, max=1).numpy()
        
        # Create a figure with two subplots
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        
        # Plot imgPre in the first subplot
        axs[0].imshow(imgPre)
        axs[0].set_title('Pre')
        axs[0].axis('off')

        # Plot imgPost in the second subplot
        axs[1].imshow(imgPost)
        axs[1].set_title('Post')
        axs[1].axis('off')

        # Display the figure
        plt.show()
