# BOUND COORDINATES MUST BE ORDERED COUNTER-CLOCKWISE

import matplotlib.pyplot as plt
import math
import utm
import numpy as np
from python_tsp.exact import solve_tsp_dynamic_programming
from python_tsp.distances import euclidean_distance_matrix
from shapely.geometry import LineString, Point, Polygon
from collections import deque


class Flight_Zone():
    def __init__(self, bound_coords: list):
        min_x, max_x, min_y, max_y, \
        self.zone_num, self.zone_let = self.get_box(bound_coords)
        self.ref_pt = (min_x, min_y)
        self.x_dim = round(max_x - min_x)
        self.y_dim = round(max_y - min_y)
        self.bd_pts = list(map(self.GPS_to_XY, bound_coords))
        self.global_path = []

    # convert gps to relative xy points using a reference utm coordinate
    def GPS_to_XY(self, gps: tuple) -> tuple:
        lat, lon = gps
        utm_xy = utm.from_latlon(lat, lon)
        x = utm_xy[0] - self.ref_pt[0]
        y = utm_xy[1] - self.ref_pt[1]
        return (x, y) 
    
    # convert relative xy back to gps
    def XY_to_GPS(self, xy: tuple) -> tuple:
        utm_x = xy[0] + self.ref_pt[0]
        utm_y = xy[1] + self.ref_pt[1]
        gps = utm.to_latlon(utm_x, utm_y, self.zone_num, self.zone_let)
        return gps 

    # return extreme utm coordinates
    def get_box(self, bound_coords: tuple) -> tuple:
        utm_init = utm.from_latlon(bound_coords[0][0], bound_coords[0][1])
        zone_num = utm_init[2]
        zone_let = utm_init[3]

        min_x = utm_init[0]
        max_x = utm_init[0]
        min_y = utm_init[1]
        max_y = utm_init[1]

        for i in range(1, len(bound_coords)):
            utm_xy = utm.from_latlon(bound_coords[i][0], bound_coords[i][1])
            min_x = min(min_x, utm_xy[0])
            max_x = max(max_x, utm_xy[0])
            min_y = min(min_y, utm_xy[1])
            max_y = max(max_y, utm_xy[1])
        return min_x, max_x, min_y, max_y, zone_num, zone_let
    

    def process_dropzone(self, drop_bds):
        drop_pts = [np.array(self.GPS_to_XY(gps)) for gps in drop_bds]
        
        delt1 = drop_pts[0] - drop_pts[len(drop_bds)-1]
        delt2 = delt1
        shortest1 = math.hypot(delt1[0], delt2[1])
        shortest2 = math.hypot(delt2[0], delt2[1])
        pair1 = (drop_pts[0], drop_pts[len(drop_bds)-1])
        pair2 = (drop_pts[0], drop_pts[len(drop_bds)-1])

        for i in range(len(drop_bds)-1):
            delta = drop_pts[i+1] - drop_pts[i]
            length = math.hypot(delta[0], delta[1])

            if length < shortest1:
                shortest2 = shortest1
                shortest1 = length
                pair2 = pair1
                pair1 = (drop_pts[i+1], drop_pts[i])
            elif length < shortest2:
                shortest2 = length
                pair2 = (drop_pts[i+1], drop_pts[i])
        
        wp1 = ((pair1[0][0] + pair1[1][0]) / 2, (pair1[0][1] + pair1[1][1]) / 2)
        wp2 = ((pair2[0][0] + pair2[1][0]) / 2, (pair2[0][1] + pair2[1][1]) / 2)
        return [wp1, wp2]


    def draw_map(self, wps, path) -> None:
        scale = max(self.x_dim, self.y_dim)
        fig, ax1 = plt.subplots(
            figsize=(14 * abs(self.x_dim/scale), 14 * abs(self.y_dim/scale))
        )
        ax1.set_title('Flight Zone')
        ax1.set_xlabel('(meters)')
        ax1.set_ylabel('(meters)')
        ax1.set_xlim(-10, self.x_dim + 10)
        ax1.set_ylim(-10, self.y_dim + 10)

        tick_scale = math.floor(math.log(min(self.x_dim, self.y_dim), 10))
        plt.xticks(np.arange(0, self.x_dim, 10**tick_scale))
        plt.yticks(np.arange(0, self.y_dim, 10**tick_scale))

        wp_arr = np.asarray(wps)
        wp_T = wp_arr.T 
        x_wp, y_wp = wp_T

        path_arr = np.asarray(path)
        path_T = path_arr.T
        x_path, y_path = path_T

        bd_arr = np.asarray(self.bd_pts)
        bd_T = bd_arr.T
        x_bd, y_bd = bd_T

        plt.plot(x_bd, y_bd, 'r-')
        plt.plot(x_path, y_path, 'b-')
        plt.plot(x_wp, y_wp, 'go')
        plt.show()


    def gen_globalpath(self, home, wps, drop_bds):
        # waypoints to cross the dropzone
        drop_pts = self.process_dropzone(drop_bds)

        # convert gps waypoints to xy
        waypts = [self.GPS_to_XY(home)]
        for gps in wps: waypts.append(self.GPS_to_XY(gps))
        waypts.extend(drop_pts)

        # set up distance matrix
        dist_matrix = euclidean_distance_matrix(np.array(waypts))
        dist_matrix[:, 0] = 0
        dist_matrix[len(waypts) - 1][len(waypts) - 1] = 0
        dist_matrix[len(waypts) - 2][len(waypts) - 1] = 0
        for i in range(1, len(waypts) - 1): 
            dist_matrix[len(waypts) - 1][i] = 1147483647
            dist_matrix[len(waypts) - 2][i] = 1147483647

        # traveling salesman to optimally order waypoints
        order, dist = solve_tsp_dynamic_programming(dist_matrix)
        
        # add in waypoints to avoid boundary
        pt_order = [waypts[order[0]]]
        waypt_order = [waypts[order[0]]]

        for i in range(len(waypts) - 1):
            path = LineString([waypts[order[i]], waypts[order[i+1]]])
            polygon = Polygon(self.bd_pts)
            polygon_ext = LineString(list(polygon.exterior.coords))
            intersections = polygon_ext.intersection(path)
            
            if not intersections.is_empty:
                print(intersections)
                bds_btwn = []
                lims = [None] * 2

                for j in range(len(self.bd_pts) - 1):
                    bd_line = LineString([self.bd_pts[j], self.bd_pts[j+1]])
                    # if the path enters the polygon at the intersection
                    if bd_line.distance(intersections.geoms[1]) < 1e-1:
                        lims[1] = self.bd_pts[j+1]
                    # if the path exits the polygon at the intersection
                    elif bd_line.distance(intersections.geoms[0]) < 1e-1:
                        lims[0] = (self.bd_pts[j])

                # repeat for final index to 0
                bd_line = LineString([self.bd_pts[len(self.bd_pts) - 1], self.bd_pts[0]])
                if bd_line.distance(intersections.geoms[1]) < 1e-1:
                    lims[1] = self.bd_pts[j+1]
                elif bd_line.distance(intersections.geoms[0]) < 1e-1:
                    lims[0] = (self.bd_pts[j])

                # search for indices of limiting bounds
                ind0 = self.bd_pts.index(lims[0])
                ind1 = self.bd_pts.index(lims[1])
                # cross product to determine if bound is above or below path
                # add distance above or below so new path does not cross over
                for k in range(ind0, ind1 - 1, -1):
                    v1 = (path.boundary.geoms[1].x - path.boundary.geoms[0].x, 
                          path.boundary.geoms[1].y - path.boundary.geoms[0].y)   # Vector 1
                    v2 = (path.boundary.geoms[1].x - self.bd_pts[k][0], 
                          path.boundary.geoms[1].y - self.bd_pts[k][1])   # Vector 2
                    xp = v1[0]*v2[1] - v1[1]*v2[0]
                    
                    if xp <= 0: 
                        bd_around = (self.bd_pts[k][0], self.bd_pts[k][1] + 10)
                    elif xp > 0: 
                        bd_around = (self.bd_pts[k][0], self.bd_pts[k][1] - 10)

                    bds_btwn.append(bd_around)
                pt_order.extend(bds_btwn)
            
            pt_order.append(waypts[order[i+1]])
            waypt_order.append(waypts[order[i+1]])

        # convert xy to gps for global path and plot
        self.global_path = deque(map(self.XY_to_GPS, pt_order))
        self.draw_map(waypt_order, pt_order)


if __name__ == '__main__':
    # competition gps coordinates
    # IMPORT: BOUND COORDINATES MUST BE ORDERED COUNTER-CLOCKWISE
    bound_coords = [
        (38.31729702009844, -76.55617670782419), 
        (38.31594832826572, -76.55657341657302), 
        (38.31546739500083, -76.55376201277696), 
        (38.31470980862425, -76.54936361414539),
        (38.31424154692598, -76.54662761646904),
        (38.31369801280048, -76.54342380058223), 
        (38.31331079191371, -76.54109648475954), 
        (38.31529941346197, -76.54052104837133), 
        (38.31587643291039, -76.54361305817427),
        (38.31861642463319, -76.54538594175376),
        (38.31862683616554, -76.55206138505936), 
        (38.31703471119464, -76.55244787859773), 
        (38.31674255749409, -76.55294546866578),
        (38.31729702009844, -76.55617670782419),
    ]
    drop_bds = [
        (38.31461655840247, -76.54516814545798),
        (38.31442098816458, -76.54523151910101),
        (38.31440638590367, -76.54394559930905),
        (38.314208221753645, -76.54400447836372)
    ]
    test_map = Flight_Zone(bound_coords)

    start = (38.316376, -76.556096)    
    wps = [
        (38.31652512851874, -76.553698306299), 
        (38.316930096287635, -76.5504102489997),
        (38.31850420404286, -76.5520175439768 ),
        (38.318084991945966, -76.54909120275754),
        (38.317170076120384, -76.54519141386767),
        (38.31453025427406, -76.5446561487259),
        (38.31534881557715, -76.54085345989367),
        (38.316679010868775, -76.54884916043693),
        (38.31859824405069, -76.54601674779236),
        (38.31659483274522, -76.54723983506672),
        (38.315736210982266, -76.5453730176419),
        (38.31603925511844, -76.54876332974675)
    ]

    test_map.gen_globalpath(start, wps, drop_bds)