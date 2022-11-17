import utm
import numpy as np
from python_tsp.distances import euclidean_distance_matrix
from python_tsp.exact import solve_tsp_dynamic_programming
from shapely.geometry import LineString, Point, Polygon
import matplotlib.pyplot as plt

# FLIGHT PLAN PARAMETERS: minimum altitude and 'normal' altitude in meters AGL
class FlightPlan():
    def __init__(self, bound_coords: list, home: tuple, min_alt=25, avg_alt=30):
        self.min_alt = min_alt 
        self.avg_alt = avg_alt

        self.ref_pt = utm.from_latlon(home[0], home[1])
        self.zone_num = self.ref_pt[2]
        self.zone_let = self.ref_pt[3]

        self.bd_pts = self.sort_counterclockwise(list(map(self.GPS_to_local, bound_coords)))
        self.bd_pts.append(self.bd_pts[0])

        self.x_dim, self.y_dim = self.get_dims()

    # convert gps to relative xy points using a reference utm coordinate
    def GPS_to_local(self, gps: tuple) -> tuple:
        utm_xy = utm.from_latlon(gps[0], gps[1])
        x = utm_xy[0] - self.ref_pt[0]
        y = utm_xy[1] - self.ref_pt[1]

        if len(gps) == 3:
            return (x, y, gps[2]) 
        else:
            return (x, y)
    
    # convert relative xy back to gps
    def local_to_GPS(self, local: tuple) -> tuple:
        utm_x = local[0] + self.ref_pt[0]
        utm_y = local[1] + self.ref_pt[1]
        gps = utm.to_latlon(utm_x, utm_y, self.zone_num, self.zone_let)
        
        if len(local) == 3:
            return (gps[0], gps[1], local[2]) 
        else:
            return gps 

    # return dimensions of boundary polygon
    def get_dims(self) -> tuple:
        min_x = self.bd_pts[0][0]
        max_x = self.bd_pts[0][0]
        min_y = self.bd_pts[0][1]
        max_y = self.bd_pts[0][1]

        for pt in self.bd_pts:
            min_x = min(min_x, pt[0])
            max_x = max(max_x, pt[0])
            min_y = min(min_y, pt[1])
            max_y = max(max_y, pt[1])

        return round(max_x - min_x), round(max_y - min_y)
    

    def process_dropzone(self, drop_bds):
        if len(drop_bds) == 0:
            return []

        drop_bd_pts = [np.array(self.GPS_to_local(gps)) for gps in drop_bds]
        delt1 = drop_bd_pts[0] - drop_bd_pts[len(drop_bds)-1]
        delt2 = delt1

        shortest1 = np.hypot(delt1[0], delt2[1])
        shortest2 = np.hypot(delt2[0], delt2[1])

        pair1 = (drop_bd_pts[0], drop_bd_pts[len(drop_bds)-1])
        pair2 = (drop_bd_pts[0], drop_bd_pts[len(drop_bds)-1])

        for i in range(len(drop_bds)-1):
            delta = drop_bd_pts[i+1] - drop_bd_pts[i]
            length = np.hypot(delta[0], delta[1])

            if length < shortest1:
                shortest2 = shortest1
                shortest1 = length
                pair2 = pair1
                pair1 = (drop_bd_pts[i+1], drop_bd_pts[i])
            elif length < shortest2:
                shortest2 = length
                pair2 = (drop_bd_pts[i+1], drop_bd_pts[i])
        
        wp1 = ((pair1[0][0] + pair1[1][0]) / 2, (pair1[0][1] + pair1[1][1]) / 2, self.min_alt)
        wp2 = ((pair2[0][0] + pair2[1][0]) / 2, (pair2[0][1] + pair2[1][1]) / 2, self.min_alt)
        return [wp1, wp2]


    def draw_map(self, wps, path) -> None:
        scale = max(self.x_dim, self.y_dim)
        fig, ax1 = plt.subplots(
            figsize=(10 * abs(self.x_dim/scale), 10 * abs(self.y_dim/scale))
        )
        ax1.set_title('Flight Plan')
        ax1.set_xlabel('(meters)')
        ax1.set_ylabel('(meters)')
        #ax1.set_xlim(-10, self.x_dim + 10)
        #ax1.set_ylim(-10, self.y_dim + 10)

        #tick_scale = np.floor(np.log(min(self.x_dim, self.y_dim), 10))
        #plt.xticks(np.arange(0, self.x_dim, 10**tick_scale))
        #plt.yticks(np.arange(0, self.y_dim, 10**tick_scale))

        wp_arr = np.asarray(wps)
        wp_T = wp_arr.T 
        x_wp, y_wp, z_wp = wp_T

        path_arr = np.asarray(path)
        path_T = path_arr.T
        x_path, y_path, z_path = path_T

        bd_arr = np.asarray(self.bd_pts)
        bd_T = bd_arr.T
        x_bd, y_bd = bd_T

        plt.plot(x_bd, y_bd, 'r-')
        plt.plot(x_path, y_path, 'b-')
        plt.plot(x_wp, y_wp, 'go')
        plt.show()


    def sort_counterclockwise(self, points, centre = None):
        if centre:
            centre_x, centre_y = centre
        else:
            centre_x, centre_y = sum([x for x,_ in points])/len(points), sum([y for _,y in points])/len(points)
            angles = [np.arctan2(y - centre_y, x - centre_x) for x,y in points]
            counterclockwise_indices = sorted(range(len(points)), key=lambda i: angles[i])
            counterclockwise_points = [points[i] for i in counterclockwise_indices]
        return counterclockwise_points


    def run_tsp(self, waypts):
        # remove altitude column from waypts
        waypt_arr = np.empty([len(waypts), len(waypts[0])])
        for i in range(len(waypts)): 
            waypt_arr[i] = np.asarray(waypts[i])
        waypt_arr = np.delete(np.array(waypts), 2, 1)
        # set up distance matrix
        dist_matrix = euclidean_distance_matrix(waypt_arr)
        dist_matrix[:, 0] = 0
        dist_matrix[len(waypts) - 1][len(waypts) - 1] = 0
        dist_matrix[len(waypts) - 2][len(waypts) - 1] = 0
        for i in range(1, len(waypts) - 1): 
            dist_matrix[len(waypts) - 1][i] = 1147483647
            dist_matrix[len(waypts) - 2][i] = 1147483647

        # traveling salesman 
        order, dist = solve_tsp_dynamic_programming(dist_matrix)
        return order, dist


    def gen_globalpath(self, wps: list, drop_bds=[]) -> list:
        # waypoints to cross the dropzone
        drop_pts = self.process_dropzone(drop_bds)

        # convert gps waypoints to xy
        waypts = [(0, 0, 0)]
        for gps in wps: waypts.append(self.GPS_to_local(gps))
        waypts.extend(drop_pts)

        # get optimal order from tsp
        order, dist = self.run_tsp(waypts)

        # pt_order: global path, waypt_order: order of "official" waypoints
        # temp_bd_pts: stores intersections between bounds and paths
        pt_order = [waypts[order[0]]]
        waypt_order = [waypts[order[0]]]
        temp_bd_pts = [pt for pt in self.bd_pts]
        
        polygon = Polygon(self.bd_pts)
        polygon_ext = LineString(list(polygon.exterior.coords))

        for i in range(len(waypts) - 1):
            path = LineString([waypts[order[i]], waypts[order[i+1]]])
            inters = polygon_ext.intersection(path)
            
            if not inters.is_empty:
                print(inters)
                # add intersections to temporary boundary
                inters0 = (inters.geoms[0].x, inters.geoms[0].y)
                inters1 = (inters.geoms[1].x, inters.geoms[1].y)
                temp_bd_pts.extend([inters0, inters1])
                # sort counterclockwise, get "marker" indices
                temp_bd_pts = self.sort_counterclockwise(temp_bd_pts)
                ind0 = temp_bd_pts.index(inters0)
                ind1 = temp_bd_pts.index(inters1)
                # always search from small to big indices
                if ind0 < ind1:
                    ind_range = range(ind0+1, ind1)
                elif ind1 < ind0:
                    ind_range = range(ind1+1, ind0)

                # compute cross product to determine if bound is above or below path
                v1 = (path.boundary.geoms[1].x - path.boundary.geoms[0].x, 
                      path.boundary.geoms[1].y - path.boundary.geoms[0].y)   # Vector 1

                for k in ind_range:
                    v2 = (path.boundary.geoms[1].x - temp_bd_pts[k][0], 
                          path.boundary.geoms[1].y - temp_bd_pts[k][1])   # Vector 2
                    xp = v1[0]*v2[1] - v1[1]*v2[0]

                    # offset path from bound point depebding on cross product, add to global path
                    if xp <= 0: 
                        bd_around = (temp_bd_pts[k][0], temp_bd_pts[k][1] + 10, self.avg_alt)
                    elif xp > 0: 
                        bd_around = (temp_bd_pts[k][0], temp_bd_pts[k][1] - 10, self.avg_alt)
                    pt_order.append(bd_around)
            
            # append to global path and order of "official" given waypoints
            pt_order.append(waypts[order[i+1]])
            waypt_order.append(waypts[order[i+1]])
        
        # rerun tsp, reorgqnize global path including boundary offsets
        order, dist = self.run_tsp(pt_order)
        global_path = [pt_order[i] for i in order]
        self.draw_map(waypt_order, global_path)
        return global_path


if __name__ == '__main__':
    # competition gps coordinates
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

    home = (38.316376, -76.556096) 
    min_alt = 25
    avg_alt = 30
    test_map = FlightPlan(bound_coords, home, min_alt, avg_alt)

    wps = [ #LLA, example wps at 30 meters AGL
        ( 38.31652512851874,   -76.553698306299, 30), 
        (38.316930096287635,  -76.5504102489997, 30),
        ( 38.31850420404286,  -76.5520175439768, 30),
        (38.318084991945966, -76.54909120275754, 30),
        (38.317170076120384, -76.54519141386767, 30),
        ( 38.31453025427406,  -76.5446561487259, 30),
        ( 38.31534881557715, -76.54085345989367, 30),
        (38.316679010868775, -76.54884916043693, 30),
        (38.315736210982266,  -76.5453730176419, 30),
        ( 38.31603925511844, -76.54876332974675, 30)
    ]

    test_map.gen_globalpath(wps, drop_bds)