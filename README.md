# Polygon Central Point Finder
Finds the point inside a polygon that minimizes the maximum distance to its border.
If the centroid is within the polygon, the centroid and the radius of the minimum enclosing circle are returned.
If the centroid is outside of the polygon, the best fitting point (with an accuracy to be set in m) + the maximum distance from the point to each polygon point are returned.
The fitting is done using an iterative grid-based approach.

## Features
- Works with polygons or multipolygons
- Works with shape files with multiple polygons
- Writes results to a shapefile (points as point geometry features + distances as attribute)

was inititally designed to derive location + uncertainty for monitoring of amphibia and reptile reports, when only addresses are reported.
was used on parcels derived from the digital cadasdral map, can be used for any polygons.
Use polygons in a projected coordinate reference system for useful results!

## Usage
```python
from finding_central_point_of_polygon import process_shapefile

input_shp = "input.shp" 	# path to your input polygon shape files
output_shp = "output.shp" 	# path to your output point shape file 
accuracy = 0.5 				# desired accuracy in m - affects the processing time

process_shapefile(input_shp, output_shp, accuracy)