from shapely.geometry import Polygon, LineString, MultiPoint
import geopandas as gpd
import numpy as np

def rel_road_lines(geodf: gpd.GeoDataFrame,
                    query_bbox_poly: Polygon,
                    res):
    """
    Given a Geodataframe containing Linestrings with geo coords, 
    returns the relative coordinates of those lines w.r.t. a reference bbox

    Inputs:
        geodf: GeoDataFrame containing the Linestring
        query_bbox_poly: Polygon of the reference bbox
        res: resolution of the image
    Returns:
        result: list of lists of tuples with the relative coordinates
    """
    ref_coords = query_bbox_poly.bounds
    ref_minx, ref_maxy = ref_coords[0], ref_coords[3] #coords of top left corner

    result = []
    for line in geodf.geometry:
        x_s, y_s = line.coords.xy

        rel_x_s = (np.array(x_s) - ref_minx) / res
        rel_y_s = (ref_maxy - np.array(y_s)) / res
        rel_coords = list(zip(rel_x_s, rel_y_s))
        line = LineString(rel_coords)
        result.append(line)
    return result

def plotLinestrings(lines, ax, color = 'red', linewidth = 1):
    """
    Plots a list of shapely linestrings
    """
    
    if not isinstance(lines, list):
        lines = [lines]

    for line in lines:
        x_s, y_s = line.coords.xy
        ax.plot(x_s, y_s, color=color, linewidth=linewidth)

def plotPoints(points, ax, color = 'red', markersize = 5):
    """
    Plots a list of shapely points
    """
    
    if not isinstance(points, list):
        points = [points]

    for point in points:
        x, y = point.xy
        ax.plot(x, y, 'o', color = color, markersize = markersize)

def line2points(lines, points_dist):
    """
    Given a list of shapely.LineString, returns a list of shapely points along all the lines, spaced by points_dist
    """
    if not isinstance(lines, list):
        lines = [lines]
    points = []
    for line in lines:
        points.extend([line.interpolate(dist) for dist in np.arange(0, line.length, points_dist)])
    return points

def get_offset_lines(lines, distance=35):
    """
    Create two offset lines from a given line at distance 'distance'
    """
    if not isinstance(lines, list):
        lines = [lines]
    
    offset_lines = []
    for line in lines:
        for side in [-1, +1]:
            offset_lines.append(line.offset_curve(side*distance ))
    return offset_lines

def clear_roads(lines, bg_points, distance):
    """
    Remove bg points that may be on the road
    """
    candidate_bg_pts = bg_points
    final_bg_pts = set(bg_points)

    if not isinstance(lines, list):
        lines = [lines]

    for line in lines:
        line_space = line.buffer(distance)
        for point in candidate_bg_pts:
            if line_space.contains(point):
                final_bg_pts.discard(point)
        
    return list(final_bg_pts)

def rmv_rnd_fraction(points, fraction_to_keep):
    """
    Removes a random fraction of the points
    """
    np.random.shuffle(points)
    points = points[:int(len(points)*fraction_to_keep)]
    return points

def rmv_pts_out_img(points: np.array, sample_size):
    """
    Removes points outside the image
    """
    points = points[np.logical_and(np.logical_and(points[:, 0] >= 0, points[:, 0] <= sample_size), np.logical_and(points[:, 1] >= 0, points[:, 1] <= sample_size))]
    return points

def segment_roads(predictor, img4Sam, road_lines, sample_size, road_point_dist = 50, bg_point_dist = 80, offset_distance = 50):
    predictor.set_image(img4Sam)
    
    final_mask = np.full((sample_size, sample_size), False)
    
    final_pt_coords4Sam = []
    final_labels4Sam = []
    
    for road in road_lines:
        road_pts = line2points(road, road_point_dist)
        
        bg_lines = get_offset_lines(road, offset_distance)
        bg_pts = line2points(bg_lines, bg_point_dist)

        bg_pts = clear_roads(road_lines, bg_pts, offset_distance - 4)

        np_roads_pts = np.array([list(pt.coords)[0] for pt in road_pts])
        np_roads_pts = rmv_pts_out_img(np_roads_pts, sample_size)
        np_road_labels = np.array([1]*np_roads_pts.shape[0])

        np_bg_pts = np.array([list(pt.coords)[0] for pt in bg_pts])
        np_bg_pts = rmv_pts_out_img(np_bg_pts, sample_size)
        np_bg_labels = np.array([0]*np_bg_pts.shape[0])

        pt_coords4Sam = np.concatenate((np_roads_pts, np_bg_pts))
        labels4Sam = np.concatenate((np_road_labels, np_bg_labels))

        final_pt_coords4Sam.extend(pt_coords4Sam.tolist())
        final_labels4Sam.extend(labels4Sam.tolist())


        if len(pt_coords4Sam) == 0 or len(labels4Sam[labels4Sam==1]) < 1 : #if there are no points, one or zero road points
            continue
        
        else:
            mask, _, _ = predictor.predict(
                    point_coords=pt_coords4Sam,
                    point_labels=labels4Sam,
                    multimask_output=False,
                )
            
            

            final_mask = np.logical_or(final_mask, mask[0])
            
    #adj_final_mask = clear_mask(road_lines, final_mask, offset_distance)
    return final_mask[np.newaxis, :], np.array(final_pt_coords4Sam), np.array(final_labels4Sam)


def clear_mask(road_lines, final_mask, offset_distance):
    
    all_true_pts = [(y,x) for x,y in zip(*np.where(final_mask))] #tutti i punti della mask final predetti come true
    all_true_multipoints = MultiPoint(all_true_pts)

    for i, line in enumerate(road_lines):
        if i == 0:
            gbl_line_space = line.buffer(offset_distance)
        else:
            gbl_line_space = gbl_line_space.union(line.buffer(offset_distance))
    
    #Seleziono tutti i punti che sono true e sono dentro gbl_line_space

    cleaned_true = all_true_multipoints.intersection(gbl_line_space)
    pt_set_false = all_true_multipoints - cleaned_true
    for pt in pt_set_false.geoms:
        final_mask[int(pt.coords[0][1]), int(pt.coords[0][0])] = False
    
    return final_mask
    

