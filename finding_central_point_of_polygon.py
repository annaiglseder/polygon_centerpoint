# -*- coding: utf-8 -*-
"""
Created on Mon Oct 27 11:19:08 2025

@author: aiglsede
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 16:42:14 2024

@author: aiglsede
"""

'''
The following code finds the central point of a polygon, which is defined as the 
point with the shortest maximum distance to any of the polygon's defining vertices 
with the addition, that the point has to be within the polygons area

The code following approach is used:
    1) Calculation of the minimum enclosing circle using Welzl Algorithm (implemented in miniball)
    2) If the point within the polygon area, the point is projected to the closest boundary line of the polygon
    3) The projected point is further used within an iterative grid approach to find the optimal point within the polygon
    
The code can handle shapefiles with polygons (Polygon and Multipolygon) and returns a shape file (Points) with the optimal 
central points, including the attributes of the initial polygons and the maximal distance of this point to any of the 
polygon's vertices.

The script is intended to be used for defining locations + uncertainties of citizen science reportings, where 
exact coordinates are not available, but a description of communal area can be extracted.

Point, Polygon, MultiPolygon, LineString refer to shapely objects

'''

import geopandas as gpd
from shapely.geometry import Polygon, Point, MultiPolygon, LineString
from shapely.ops import unary_union
import shapely as shp
import miniball
import math
import numpy as np
from tqdm import tqdm
import os


def compute_min_enclosing_circle(polygon):
    '''
    Computes the minimun enclosing cirlce of a polygon based on Welzl's Algorithm implemented in miniball
    
    Parameters:
        - polygon: Polygon or MultiPolygon
    
    Returns:
        - center: Point
        - radius: radius of the enclosing circle as float
        
    '''

    # If the input is a MultiPolygon, merge the vertices from all polygons
    if isinstance(polygon, MultiPolygon):
        coords = []
        for poly in polygon.geoms:
            coords.extend(list(poly.exterior.coords))
    elif isinstance(polygon, Polygon):
        # For a single Polygon, use its exterior coordinates
        coords = list(polygon.exterior.coords)
    else:
        # Log unsupported geometry attributes if the input is neither Polygon nor MultiPolygon
        raise ValueError(f"Unsupported geometry type: {type(polygon)}")

    # Remove duplicate points
    coords = list(set(coords))
    coords = np.array(coords)

    # Compute the minimal enclosing circle
    center, radius_square = miniball.get_bounding_ball(coords)
    radius = math.sqrt(radius_square)

    return center, radius



def create_grid_within_polygon(polygon, spacing, area_polygon=None, step_fraction=2):
    """
    Create a grid of points within the polygon with the specified spacing,
    optionally limiting the area to a smaller polygon.

    Parameters:
       - polygon: initial Polygon or MultiPolygon 
       - spacing (float): The spacing between grid points.
       - area_polygon: Smaller Polygon limiting the grid area.

    Returns:
        - points (list): A list of Point objects within the specified area.
    """
    # Get the bounding box of the polygon
    search_area = area_polygon if area_polygon else polygon
    minx, miny, maxx, maxy = search_area.bounds
    
    # Generate grid points
    step = float(spacing) / float(step_fraction)
    xs = np.arange(minx, maxx + step, step, dtype=float)
    ys = np.arange(miny, maxy + step, step, dtype=float)
    xx, yy = np.meshgrid(xs, ys)
    grid_xy = np.column_stack([xx.ravel(), yy.ravel()])
    
    pts = shp.points(grid_xy[:, 0], grid_xy[:, 1])
    mask = shp.covers(search_area, pts)
    mask = np.asarray(mask, dtype=bool)
    
    return grid_xy[mask]
    
    

def get_bounding_polygon_around_point(point, spacing):
    """
    Get a bounding polygon (square) around the specified point using the full grid spacing.

    Parameters:
        - point (Point): The center point.
        - spacing (float): Distance to the neighbors (full grid spacing).

    Returns:
        - bounding_polygon (Polygon): Polygon bounding the 8 neighbors.
    """
    x, y = point.x, point.y

    # Define the square's vertices (full spacing around the point)
    vertices = [
        (x - spacing, y - spacing),
        (x + spacing, y - spacing),
        (x + spacing, y + spacing),
        (x - spacing, y + spacing)
    ]
    return Polygon(vertices)


def min_of_max_distance(grid_xy, hull_xy, mem_limit_bytes=2*1024**3, safety=0.80, dtype=None):
    """
    Find the grid point minimizing the maximum distance to all hull vertices.
    Batch size is chosen automatically so peak working memory for (B×N) arrays
    stays under ~mem_limit_bytes * safety.

    Returns
    -------
    best_point : shapely.geometry.Point
    best_dist  : float
    """
    if grid_xy.size == 0 or hull_xy.size == 0:
        return Point(np.nan, np.nan), float("nan")

    # Decide computation dtype
    if dtype is None:
        # default to float64 to match typical Shapely coords
        dtype = np.float64
    grid_xy = np.asarray(grid_xy, dtype=dtype, order="C")
    hull_xy = np.asarray(hull_xy, dtype=dtype, order="C")

    M = grid_xy.shape[0]
    N = hull_xy.shape[0]
    itemsize = np.dtype(dtype).itemsize

    # Working memory model: two (B×N) arrays (d2 and tmp) in RAM simultaneously
    # bytes_needed ≈ 2 * B * N * itemsize  <= budget
    budget = int(mem_limit_bytes * safety)
    denom = 2 * N * itemsize
    if denom <= 0:
        denom = 1
    B = max(1, min(M, budget // denom))

    # Failsafe: if N is *so* large that B becomes 0 due to tiny budget, keep B=1
    if B < 1:
        B = 1

    best_idx_global = None
    best_dist2_global = np.inf

    Hx = hull_xy[:, 0]
    Hy = hull_xy[:, 1]

    for i0 in range(0, M, B):
        i1 = min(i0 + B, M)
        G = grid_xy[i0:i1]  # (B,2)

        Gx = G[:, 0]
        Gy = G[:, 1]

        # In-place, low-temp approach:
        d2 = Gx[:, None] - Hx[None, :]      # (B,N)
        np.square(d2, out=d2)               # d2 = (Gx-Hx)^2

        tmp = Gy[:, None] - Hy[None, :]     # (B,N)
        np.square(tmp, out=tmp)             # tmp = (Gy-Hy)^2
        d2 += tmp                           # d2 = dx^2 + dy^2

        max_d2 = np.max(d2, axis=1)         # (B,)
        j_local = int(np.argmin(max_d2))
        d2_best_local = float(max_d2[j_local])

        if d2_best_local < best_dist2_global:
            best_dist2_global = d2_best_local
            best_idx_global = i0 + j_local

        # Let tmp and d2 go out of scope here before next batch

    best_xy = grid_xy[best_idx_global]
    best_point = Point(float(best_xy[0]), float(best_xy[1]))
    best_dist = float(np.sqrt(best_dist2_global))
    return best_point, best_dist

def hull_vertices_xy(geom):
    """
    Return an (N,2) array of convex-hull vertices (exterior only), without the
    duplicate closing point. Handles Polygon, MultiPolygon, and degeneracies.
    """
    if isinstance(geom, MultiPolygon):
        geom = unary_union(geom)  # merge parts before hull

    hull = geom.convex_hull
    # Degenerate hulls can be Point or LineString
    if isinstance(hull, Point):
        xy = np.array([[hull.x, hull.y]], dtype=float)
    elif isinstance(hull, LineString):
        xy = np.asarray(hull.coords, dtype=float)
    else:
        # Polygon hull
        xy = np.asarray(hull.exterior.coords[:-1], dtype=float)  # drop duplicate close
    return xy

def border_vertices_with_spacing(geom, max_seg, min_spacing=0.0, dtype=float):
    """
    Extract exterior border vertices and insert evenly spaced intermediate points
    on edges longer than 'max_seg'. Removes points closer than 'min_spacing'.
    Ignores holes. Works for Polygon or MultiPolygon.

    Parameters
    ----------
    geom : shapely Polygon or MultiPolygon
        Input geometry (projected units).
    max_seg : float
        Maximum allowed edge length (edges longer than this are subdivided).
    min_spacing : float, optional
        Minimum allowed spacing between consecutive points (shorter gaps removed).
    dtype : data type, optional
        Output dtype (default: float)

    Returns
    -------
    xy : (N,2) ndarray
        Cleaned border coordinates (no duplicate closing point).
    """
    if max_seg <= 0:
        raise ValueError("max_seg must be > 0")
    if min_spacing < 0:
        raise ValueError("min_spacing must be >= 0")

    def _ring_coords_with_intermediates(ring):
        coords = np.asarray(ring.coords, dtype=dtype)[:-1]  # drop duplicate closing vertex
        if len(coords) == 0:
            return np.empty((0, 2), dtype=dtype)

        out = [coords[0]]
        for i in range(1, len(coords)):
            p0 = coords[i-1]
            p1 = coords[i]
            vec = p1 - p0
            L = float(np.hypot(vec[0], vec[1]))
            if L > max_seg:
                k = int(np.floor(L / max_seg))
                ts = np.linspace(0, 1, num=k+2, endpoint=True)[1:-1]
                for t in ts:
                    out.append(p0 + t * vec)
            out.append(p1)
        out = np.vstack(out)

        # Remove consecutive points that are too close
        if min_spacing > 0 and len(out) > 1:
            keep = [True]
            last = out[0]
            for i in range(1, len(out)):
                if np.hypot(out[i,0]-last[0], out[i,1]-last[1]) >= min_spacing:
                    keep.append(True)
                    last = out[i]
                else:
                    keep.append(False)
            out = out[np.array(keep, dtype=bool)]

        return out

    parts = []
    if isinstance(geom, Polygon):
        parts.append(_ring_coords_with_intermediates(geom.exterior))
    elif isinstance(geom, MultiPolygon):
        for p in geom.geoms:
            parts.append(_ring_coords_with_intermediates(p.exterior))
    else:
        raise ValueError("Geometry must be Polygon or MultiPolygon")

    if not parts:
        return np.empty((0, 2), dtype=dtype)

    xy = np.vstack(parts)

    # Remove accidental consecutive duplicates again (edge-case safety)
    if len(xy) > 1:
        diff = np.diff(xy, axis=0)
        mask = np.any(diff != 0, axis=1)
        xy = np.vstack([xy[0], xy[1:][mask]])

    return xy

def choose_initial_spacing(
    polygon,
    min_spacing,
    step_fraction=2,
    N_target=8000,
    N_min=2000,
    N_max=20000,
    k0=16,
    clamp_diag_frac=8.0   # upper clamp = bbox_diag / clamp_diag_frac
):
    """
    Choose a size-aware initial 'spacing' for your grid routine.

    Parameters
    ----------
    polygon : shapely Polygon/MultiPolygon
    min_spacing : float
        Target accuracy of the final search (meters). Initial spacing will be >> this.
    step_fraction : int
        Your grid function uses step = spacing / step_fraction.
    N_target : int
        Desired initial grid point count.
    N_min, N_max : int
        Acceptable band for initial grid point count (will adjust spacing to fall in here).
    k0 : float
        Coarseness factor relative to min_spacing (spacing >= k0 * min_spacing).
    clamp_small : float
        Smallest allowed spacing is clamp_small * min_spacing.
    clamp_diag_frac : float
        Largest allowed spacing is bbox_diag / clamp_diag_frac.

    Returns
    -------
    spacing : float
        Initial spacing to pass to create_grid_within_polygon(...).
    """
    # geometry measures
    area = polygon.area
    minx, miny, maxx, maxy = polygon.bounds
    bbox_diag = float(np.hypot(maxx - minx, maxy - miny))

    # 1) accuracy-based coarse start
    s_acc = k0 * float(min_spacing)

    # 2) size-based to hit ~N_target points (for square grid with step=spacing/step_fraction)
    if area <= 0:
        s_cnt = s_acc
    else:
        step_target = np.sqrt(area / float(N_target))
        s_cnt = step_target * float(step_fraction)

    # combine (pick the larger so we don't start too fine)
    spacing = max(s_acc, s_cnt)
    
    # check to not be too large
    s_max = bbox_diag / float(clamp_diag_frac) if bbox_diag > 0 else spacing    
    spacing = min(spacing, s_max)

    # 3) adjust to ensure initial point count within [N_min, N_max]
    # estimated count with effective step
    def est_count(sp):
        st = sp / float(step_fraction)
        if st <= 0:
            return np.inf
        return area / (st * st)

    N_est = est_count(spacing)

    if N_est > N_max:
        # too many points -> increase spacing
        factor = np.sqrt(N_est / float(N_target))
        spacing = min(spacing * factor, s_max)
    elif N_est < N_min:
        # too few points -> decrease spacing
        factor = np.sqrt(float(N_target) / max(N_est, 1e-9))
        spacing = max(spacing / factor, s_acc)

    return float(spacing)

def find_optimal_point_iterative(polygon, min_spacing, plot=False):
    """
    Iteratively refine the grid to find the optimal point inside the polygon,
    limiting the search area to neighbors of the current optimal point.

    Parameters:
    - polygon (Polygon): The polygon geometry.
    - initial_spacing (float): Initial grid spacing in meters.
    - min_spacing (float): Minimum grid spacing for refinement.

    Returns:
    - optimal_point (Point): The point inside the polygon that minimizes the maximum distance.
    - min_max_distance (float): The minimum maximum distance to the border vertices.
    """
    
    spacing = choose_initial_spacing(polygon, min_spacing)
    optimal_point = None
    min_max_distance = float('inf')
    search_area = polygon  # Initially the entire polygon
    hull_vertices = hull_vertices_xy(polygon)
    polygon_vertices = border_vertices_with_spacing(polygon, spacing/2, min_spacing)       
    
    while spacing >= min_spacing:
        # Generate grid points within the polygon or limited area
        grid_points = create_grid_within_polygon(polygon, spacing, search_area)
        if polygon_vertices is not None:
            grid_points = np.vstack([grid_points, polygon_vertices])
        # Evaluate grid points
        optimal_point, min_max_distance = min_of_max_distance(grid_points, hull_vertices)
               
        # Refine the grid spacing and limit the search area
        search_area = get_bounding_polygon_around_point(optimal_point, spacing)
        search_area = search_area.intersection(polygon)  # Ensure it stays within the polygon
        spacing /= 2
        polygon_vertices = None
        

    return optimal_point, min_max_distance


def process_shapefile(input_shapefile, output_shapefile, accuracy=0.1):
    """
    Processes the input shapefile, calculating optimal points based on whether the center is inside
    or outside the polygon. Outputs a new shapefile with the results and displays a progress bar.
    
    Parameters:    
        - input_shapefile: Path to the input shapefile
        - output_shapefile: Path to save the output shapefile
        - initial_spacing: initial spacing of the grids
    """
    # Load the shapefile
    gdf = gpd.read_file(input_shapefile)
    
    # CRS check
    
    if gdf.crs is None:
        raise ValueError("Input shapefile has no CRS defined. Assign a projected CRS (e.g., EPSG:31256).")

    if gdf.crs.is_geographic:
        raise ValueError(
            "\n\n ################## \n"
            f"\n The input CRS ({gdf.crs.to_string()}) is geographic (lat/lon). \n"
            "This script requires a projected CRS in meters (e.g., UTM or EPSG:31256).\n"
            "\n ##################"
        )
        
    # Geometry validity check
    
    invalid_mask = ~gdf.is_valid
    
    if invalid_mask.any():
        n_invalid = invalid_mask.sum()
        
        raise ValueError(
            "\n\n ################## \n"
             f"The shapefile contains {n_invalid} invalid geometries "
             "(self-intersections, bowties, etc.). "
             "Fix them in QGIS or with GeoPandas: gdf = gdf.buffer(0)."
             "\n ##################"
            )   

    # Geometry type check
    
    allowed_types = {"Polygon", "MultiPolygon"}
    geom_types = set(gdf.geom_type.unique())
    
    if not geom_types.issubset(allowed_types):
        raise ValueError(
            f"The shapefile contains unsupported geometry types: {geom_types}. "
            "Only Polygon and MultiPolygon geometries are supported."
        )                
    
        
    
    total_rows = len(gdf)  # Total number of rows for the progress bar
    results = [] 

    # Iterate through rows with a progress bar
    for idx, row in tqdm(gdf.iterrows(), total=total_rows, desc="Processing rows", unit="row"):
        #selected_idx = 0
        #row = gdf.iloc[selected_idx]
        try:
            polygon = row['geometry']

            # Calculate the minimum enclosing circle
            center_coords, radius = compute_min_enclosing_circle(polygon)
            center_point = Point(center_coords)

            # Test if the point is inside the polygon or on the boundary
            is_inside = polygon.covers(center_point)

            if is_inside:
                # Create a new database entry for the resulting shapefile
                result = row.copy()
                result['geometry'] = center_point
                result['uncert'] = radius
                result['center_inside'] = is_inside
                results.append(result)

            else:

                optimal_point, min_max_distance = find_optimal_point_iterative(polygon, accuracy)

                result = row.copy()
                result['geometry'] = optimal_point
                result['uncert'] = min_max_distance
                result['center_inside'] = is_inside
                results.append(result)

        except Exception as e:
            print(f"Error processing row {idx}: {e}")
            continue

    # Convert results to a GeoDataFrame
    result_gdf = gpd.GeoDataFrame(results, crs=gdf.crs)

    # Save the output shapefile
    result_gdf.to_file(output_shapefile)
    print(f"Processing complete. Results saved to {output_shapefile}")



'''

TESTING


wd = ## path to working directory


# Paths to input and output shapefiles
input_shapefile = os.path.join(wd, "polygon.shp")
output_shapefile = os.path.join(wd, "polygon_result.shp")
#

import time

start = time.time()

# Run the processing function
process_shapefile(input_shapefile, output_shapefile, accuracy=0.01)

end = time.time()
elapsed = end-start

print(f"processing took {elapsed} seconds.")

'''
