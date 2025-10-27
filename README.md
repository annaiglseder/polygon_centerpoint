# Polygon Central Point Finder
Finds the point inside a polygon that minimizes the maximum distance to its border vertices.
If the centroid is within the polygon, the centroid and the radius of the minimum enclosing circle are returned.
If the centroid is outside of the polygon, the best fitting point (with an accuracy to be set) + the maximum distance from the point to each polygon point are returned

## Features
- Works with polygons or multipolygons
- Works with shape files with multiple polygons
- Writes results to a shapefile (points as point geometry features + distances as attribute)

was inititally designed to derive location + uncertainty for monitoring of amphibia and reptile reports, when only addresses are reported.
was used on parcels derived from the digital cadasdral map

## Usage
```python
from polygon_center import process_shapefile
process_shapefile("input.shp", "output.shp", min_spacing=1.0)