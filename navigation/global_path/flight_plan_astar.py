import matplotlib.pyplot as plt
import math
import utm
import numpy as np
from astar import astar


class FlightPlan():
    def __init__(self, bound_coords: list):
        min_x, max_x, min_y, max_y, \
        self.zone_num, self.zone_let = self.get_box(bound_coords)
        self.ref_pt = (min_x, min_y)
        self.x_dim = round(max_x - min_x)
        self.y_dim = round(max_y - min_y)
        self.boundary = self.calc_boundary(bound_coords)
        self.wp_order = []

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
    

    def calc_boundary(self, bound_coords) -> list:
        bd_pts = list(map(self.GPS_to_XY, bound_coords))
        bd_pts = [(round(pt[0]), round(pt[1])) for pt in bd_pts]
        disc_bdry = []
        
        for i in range(0, len(bd_pts)-1, 1):
            Xo = bd_pts[i][0]
            X1 = bd_pts[i+1][0]
            Yo = bd_pts[i][1]
            Y1 = bd_pts[i+1][1]
            m = (Y1 - Yo)/(X1 - Xo) 
            step = 1

            if abs(X1 - Xo) >= abs(Y1 - Yo):
                if X1 - Xo < 0:
                    step = -1
                for x in range(Xo, X1, step):
                    y = round(float(m * (x - Xo) + Yo))
                    disc_bdry.append((x, y)) 
            else:
                if Y1 - Yo < 0:
                    step = -1
                for y in range(Yo, Y1, step):
                    x = round(float((y - Yo)/m + Xo))
                    disc_bdry.append((x, y)) 
        return disc_bdry


    def draw_map(self, wps, path) -> None:
        scale = max(self.x_dim, self.y_dim)
        fig, ax1 = plt.subplots(
            figsize=(10 * abs(self.x_dim/scale), 10 * abs(self.y_dim/scale))
        )
        ax1.set_title('Flight Plan')
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

        x_vals = [pt[0] for pt in self.boundary]
        y_vals = [pt[1] for pt in self.boundary]

        plt.plot(x_vals, y_vals, 'r-')
        plt.plot(x_path, y_path, 'b-')
        plt.plot(x_wp, y_wp, 'go')
        plt.show()
    

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


    def gen_globalpath(self, home, wps, drop_bds=[]):
        # convert to xy
        home = self.GPS_to_XY(home)
        waypts = [self.GPS_to_XY(gps) for gps in wps]
        # waypoints to cross the dropzone
        if len(drop_bds):
            drop_pts = self.process_dropzone(drop_bds)
        else:
            drop_pts = drop_bds
            
        order = [home]
        curr = home
        # order the waypoints
        while len(waypts):
            nxt = waypts[0]
            idx = 0
            dist = math.hypot(nxt[0]-curr[0], nxt[1]-curr[1])

            for i in range(1, len(waypts)):
                dx = waypts[i][0] - curr[0]
                dy = waypts[i][1] - curr[1]
                temp_dist = math.hypot(dx, dy)
                drop_dist = temp_dist

                if temp_dist < dist:
                    nxt = waypts[i]
                    idx = i
                    dist = temp_dist
            
            drop_dist = dist
            if len(drop_pts):
                dx1 = drop_pts[0][0] - curr[0]
                dy1 = drop_pts[0][1] - curr[1]
                dx2 = drop_pts[1][0] - curr[0]
                dy2 = drop_pts[1][1] - curr[1]

                if  math.hypot(dx2, dy2) < math.hypot(dx1, dy1):
                    drop_pts = [drop_pts[1], drop_pts[0]]
                    drop_dist = math.hypot(dx2, dy2)
                else:
                    drop_dist = math.hypot(dx1, dy1)
            
            if drop_dist < dist:
                curr = drop_pts[1]
                order.extend(drop_pts)
                drop_pts.clear()
            else:
                curr = nxt
                order.append(nxt)
                waypts.pop(idx)

        if len(drop_pts):
            order.extend(drop_pts)
            drop_pts.clear()
        
        global_path = []
        for i in range(len(order) - 1):
            start = (round(order[i][0]), round(order[i][1]))
            end = (round(order[i+1][0]), round(order[i+1][1]))
            path = astar(self.x_dim, self.y_dim, self.boundary, start, end)
            global_path.extend(path)
        
        self.wp_order = list(map(self.XY_to_GPS, order))
        print('waypoint order:')
        print(self.wp_order)
        self.draw_map(order, global_path)


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
        (38.314208221753645, -76.54400447836372),
        (38.31461655840247, -76.54516814545798)
    ]
    test_map = FlightPlan(bound_coords)

    start = (38.316376, -76.556096)    
    wps = [
        (38.31652512851874, -76.553698306299), 
        (38.316930096287635, -76.5504102489997),
        (38.318084991945966, -76.54909120275754),
        (38.317170076120384, -76.54519141386767),
        (38.31453025427406, -76.5446561487259),
        (38.31534881557715, -76.54085345989367)
    ]
    test_map.gen_globalpath(start, wps, drop_bds)
    #test_map.draw_map(w, h, path)