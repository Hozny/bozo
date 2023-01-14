from scipy.spatial import Delaunay
import matplotlib.pyplot as plt
import numpy as np
from itertools import chain
import math
from scipy.spatial import Voronoi, voronoi_plot_2d

# points = np.array([[0, 0], [0, 0.5], [1, 0], [1, 1]])

def calculate_polygons(startx, starty, endx, endy, radius):
     # calculate side length given radius   
    sl = (2 * radius) * math.tan(math.pi / 6)
    # calculate radius for a given side-length
    # (a * (math.cos(math.pi / 6) / math.sin(math.pi / 6)) / 2)
    # see http://www.calculatorsoup.com/calculators/geometry-plane/polygon.php
    
    # calculate coordinates of the hexagon points
    # sin(30)	
    p = sl * 0.5
    b = sl * math.cos(math.radians(30))
    w = b * 2
    h = 2 * sl
    
    # offset start and end coordinates by hex widths and heights to guarantee coverage     
    startx = startx - w
    starty = starty - h
    endx = endx + w
    endy = endy + h

    origx = startx
    origy = starty


    # offsets for moving along and up rows
    xoffset = b
    yoffset = 3 * p

    polygons = []
    row = 1
    counter = 0

    while starty < endy:
        if row % 2 == 0:
            startx = origx + xoffset
        else:
            startx = origx
        while startx < endx:
            p1x = startx
            p1y = starty + p
            p2x = startx
            p2y = starty + (3 * p)
            p3x = startx + b
            p3y = starty + h
            p4x = startx + w
            p4y = starty + (3 * p)
            p5x = startx + w
            p5y = starty + p
            p6x = startx + b
            p6y = starty
            poly = [
                (p1x, p1y),
                #(p2x, p2y),
                (p3x, p3y),
                #(p4x, p4y),
                (p5x, p5y),
                #(p6x, p6y),
                (p1x, p1y)]
            polygons.append(poly)
            counter += 1
            startx += w
        starty += yoffset
        row += 1
    return polygons

pts = np.array(calculate_polygons(0, 0, 5, 5, 1))

pts = np.array(list(chain(*pts)))
tri = Delaunay(pts)

vor = Voronoi(pts)
print(vor.vertices)
print(vor.regions)
fig = voronoi_plot_2d(vor)
plt.show()

# plt.triplot(pts[:,0], pts[:,1], tri.simplices)
# plt.plot(pts[:,0], pts[:,1], 'o')
# plt.show()


