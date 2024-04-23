import argparse
import torch

from maxarseg.assemble import names
from maxarseg.assemble import holders
from maxarseg.configs import Config

torch.set_float32_matmul_precision('medium')

def main(): 
    events_names = names.get_all_events()    
    parser = argparse.ArgumentParser(description='Segment Maxar Tiles')
    parser.add_argument('--config', required=True type = str, help='Path to the custom configuration file')
    parser.add_argument('--event_ix', type = int, help='Index of the event in the list events_names')
    parser.add_argument('--out_dir_root', help='output directory root')

    args = parser.parse_args()
    
    cfg = Config(config_path = args.config)
    
    if args.event_ix is not None:
        cfg.set('event/ix', args.event_ix)
        
    if args.out_dir_root is not None:
        cfg.set('output/out_dir_root', args.out_dir_root)
        
    print("Selected Event: ", events_names[args.event_ix])
    
    # check if there is cuda, otherwise use cpu
    if not torch.cuda.is_available():
        args.device_det = 'cpu'
        args.device_seg = 'cpu'
    
    
    event = holders.Event(events_names[args.event_ix], cfg = cfg)
    
    all_mosaics_names = event.all_mosaics_names
    
    event.seg_all_mosaics(out_dir_root=args.out_dir_root) #this segment all the mosiacs in the event
    
    # m0 = event.mosaics[all_mosaics_names[0]]
    # m0.segment_all_tiles(out_dir_root=args.out_dir_root) #this segment all tiles in the mosaic
    
    # m0_tile_17_path = m0.tiles_paths[17]
    # tile_path = '/nfs/projects/overwatch/maxar-segmentation/maxar-open-data/Gambia-flooding-8-11-2022/pre/105001002BD68F00/033133031213.tif'
    # m0.segment_tile(tile_path, args.out_dir_root, separate_masks = False)

if __name__ == "__main__":
    main()