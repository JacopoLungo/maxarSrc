from torchgeo.datasets import RasterDataset, IntersectionDataset
from torchgeo.datasets import BoundingBox
import torch
import matplotlib.pyplot as plt
import os



#TODO: pulire il codice sotto dai commenti
class Maxar(RasterDataset):
    filename_glob = "*.tif"
    is_image = True
    separate_files = False

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
        image = sample["image"].permute(1, 2, 0)
        image = torch.clamp(image / 300, min=0, max=1).numpy()

        # Plot the image
        fig, ax = plt.subplots()
        ax.imshow(image)

        return fig

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
