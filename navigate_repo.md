## Important folders:
 * metadata
 * models
 * notebooks
 * src

### Inside notebooks:
### build_resources
* build_all: its the test notebook to be converted in a script
* explore_roads and its copy: are not important

all_mask: old method to segment a patch using SAM

## Inside src/maxarseg
## build.py

### explore_folders.py
It contains functions to explore the data folder and check if all data are presents. Plus, it can be used to compute some stats about single events.

### geoDatasets.py
It contains two dataset type:
* Maxar: to sample for a unique temporal dimension (only pre or only post)
* MaxarIntersectionDataset: to sample in the same coordinates in pre and post simultaneously

### plotting_utils.py
Contains plotting functions useful in debug phase. 

### samplers_utils.py

### samplers.py
It contains various type of samplers. The most important in the dataset creation phase is *MyBatchGridGeoSampler*.

### segment.py

Build - mosaic class
La classe mosaico è in sostanza una collezione di path ai tile che lo compongono.

Una volta costruito il gdf degli edifici e delle strade del mosaico puoi segmentarlo.

Il metodo get_tile_road_mask_np ti restituisce la mask delle strade di tutto il tile come np array.

Il metodo segment_tile invece funziona dividendo l'intero tile in varie patches.
Viene creato un Maxar dataset (che contiene un solo tile) e vengono estratte batch (b) immagini ogni volta che viene preso un sample
