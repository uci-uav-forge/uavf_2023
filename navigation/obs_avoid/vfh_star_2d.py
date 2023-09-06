from dataclasses import dataclass
import numpy as np
import math
import heapq

@dataclass
class VFHParams:
    D_t: float # total distance in meters planned per VFH iteration
    N_g: int # number of steps to plan out at a time
    R: float # steering radius: left/right turns
    alpha: int # angular resolution: how many windows to divide 180 degrees into?
    b: float # controls how fast increased distance from sensor affects confidence
    mu_1: float # weight penalizing candidate direction going away from target
    mu_2: float # weight penalizing turning from current direction
    mu_3: float # weight penalizing turning from previous selected direction
    # (paper recommends mu_1 > mu_2 + mu_3)
    mu_1p: float
    mu_2p: float
    mu_3p: float
    # above parameters used to calculate cost for projected future motion rather than immediate plan
    lmbda: float # discount factor for planned positions
    hist_res: float # width in meters of a histogram block

    r_active: float # distance in meters to examine for polar histogram (active window radius)
    r_drone: float # distance to consider our width as for widening obstacles in polar histogram (includes safety distance)

    t_low: float # low threshold for binary histogram
    t_high: float # high threshold

    mask_conf: float

    dir_spacing: int # how much to space out directions


class VFH2D:
    def __init__(self, params: VFHParams, hist: 'Histogram'):
        self.params = params
        self.hist = hist

    def pos_to_idx(self, pos: np.array) -> np.array:
        return np.round(pos/self.params.hist_res)
    
    def gen_polar_histogram(self, pos: np.array) -> np.ndarray:
        result = np.zeros(2*self.params.alpha)

        a = self.params.b * (self.params.r_active)**2
        # note: paper wants above to be a = 1 + b(r_active -1)**2/4
        # this gives weird negative results for far things ... idk what it's doing.

        idx = self.pos_to_idx(pos)
        di = math.ceil(self.params.r_active / self.params.hist_res)
        for dx in range(-di,di+1):
            mx = (dx+0.5)*self.params.hist_res
            dy_max = math.ceil(max(0,self.params.r_active**2 - mx**2)**0.5/self.params.hist_res)
            for dy in range(-dy_max, dy_max):
                    my = (dy+0.5)*self.params.hist_res
                    dist = (mx**2 + my**2)**0.5
                    if self.params.r_drone <= dist <= self.params.r_active:
                        idx2 = idx + np.array([dx,dy,0])
                        conf = self.hist.get_confidence(idx2)

                        theta = math.atan2(dy,dx)

                        enlargment_angle = math.asin(self.params.r_drone/dist)

                        theta_idx = round((theta/(2*math.pi))*2*self.params.alpha)

                        enlarge_idx = math.ceil(enlargment_angle/(math.pi)*self.params.alpha)

                        for tidx in range(theta_idx - enlarge_idx, theta_idx + enlarge_idx+1):
                            result[tidx % (2*self.params.alpha)] += conf*conf*(a - self.params.b*dist*dist)
        return result
    
    def gen_bin_histogram(self, polar_hist: np.ndarray) -> np.ndarray:
        results = np.zeros(2 * self.params.alpha)
        for j in range(self.params.alpha * 2):
            if polar_hist[j] > self.params.t_high:
                results[j] = 1
            elif polar_hist[j] < self.params.t_low:
                results[j] = 0
            else:
                # for now 
                results[j] = 1 
        return results


    def gen_masked_histogram(self, bin_hist: np.ndarray, pos: np.array, theta: float) -> np.ndarray:
        results = bin_hist.copy()

        # left and right displacements to mask out for each phi
        thetas = [0, 0]

        lx = pos[0] + self.params.R*math.cos(theta)
        ly = pos[1] + self.params.R*math.sin(theta)
        
        rx = pos[0] - self.params.R*math.cos(theta)
        ry = pos[1] - self.params.R*math.sin(theta)

        idx = self.pos_to_idx(pos)
        di = math.ceil(self.params.r_active / self.params.hist_res)
        # minor todo: iterator method for this pattern.
        for dx in range(-di,di+1):
            for dy in range(-di, di+1):
                mx = (dx+0.5)*self.params.hist_res
                my = (dy+0.5)*self.params.hist_res
                dist = (mx**2 + my**2)**0.5
                if not (self.params.r_drone <= dist <= self.params.r_active):
                    continue
                
                theta_pos = math.atan2(dy,dx)

                idx2 = idx + np.array([dx,dy,0])

                if self.hist.get_confidence(idx2) > self.params.mask_conf:

                    if ((mx-lx)**2+(my-ly)**2)**0.5 < (self.params.R + self.params.r_drone):
                        dist = ((theta + math.pi) - theta_pos) % 2*math.pi
                        if dist < math.pi:
                            thetas[0] = max(thetas[0], dist)

                    
                    if ((mx-rx)**2+(my-ry)**2)**0.5 < (self.params.R + self.params.r_drone):
                        dist = (theta_pos - (theta + math.pi)) % 2*math.pi 
                        if dist < math.pi:
                            thetas[1] = max(thetas[1], dist)
        
        lt,rt = thetas
        for j in range(self.params.alpha * 2):
            theta_here = j * math.pi / self.params.alpha

            # check if it's in the range [-theta - left_theta, -theta + right_theta]
            
            dist = (theta_here - (theta+math.pi - lt)) % 2*math.pi

            if dist + 0.01 <= lt + rt:
                # block it out
                results[j] = 1
        return results


    def j2step(self, j:int):
        ja = j/(2*self.params.alpha) * 2*math.pi
        step_dist = self.params.D_t / self.params.N_g
        return np.array([step_dist*math.cos(ja), step_dist*math.sin(ja), 0])
    

    def gen_directions(self, masked_hist: np.ndarray, delta_position: np.ndarray):
        theta_dpos = math.atan2(delta_position[1],delta_position[0])

        dpos_j = round(theta_dpos * 2*self.params.alpha / (2*math.pi))

        # idea: sample in the target direction and spaced-out samples on borders of areas
        result = []
        too_close = [False] * (2*self.params.alpha)
        
        
        for j in range(self.params.alpha * 2):
            if not masked_hist[j] and not too_close[j]:
                aijs = [j-1,j+1]
                if any(masked_hist[aj % (2*self.params.alpha)] for aj in aijs):
                    result.append(j)
                    for dj in range(-self.params.dir_spacing,self.params.dir_spacing+1):
                        too_close[(j+dj) % (2*self.params.alpha)] = True
        
        if not masked_hist[dpos_j] and not too_close[dpos_j]:
            result.append(dpos_j)
        
        return [math.pi / self.params.alpha * thi for thi in result]

    def angle_dist(self, tp1, tp2):
        ad_inner = lambda a1, a2: min(abs(a1 - a2), abs(a1 - a2 + 2*math.pi), abs(a1 - a2 - 2*math.pi))
        return ad_inner(tp1, tp2)
    
    def theta_to_dxyz(self, theta):
        return np.array([math.cos(theta), math.sin(theta), 0])
    
    def dxyz_to_theta(self, dxyz):
        theta = math.atan2(dxyz[1],dxyz[0])
        return theta


    def get_target_dir(self, position: np.ndarray, theta: float, target_position: np.ndarray) -> tuple[float, float]:
        # execute A* algorithm.
        # idea: repeatedly expand nodes based on heuristic until we reach a node that has been expanded N_g times.
        # the first step on the way to that node is the target.

        # queue: tuples of cost, state, depth

        visit_queue = [(0, None, (position, theta), 0, None)]

        theta_end = self.dxyz_to_theta(target_position - position)

        unique_ctr = 0
        
        while len(visit_queue):
            cost, _, node_info, node_depth, start_dir = heapq.heappop(visit_queue)
            p_node, theta_node = node_info
            if node_depth == self.params.N_g:
                return start_dir
            
            # otherwise expand node and insert into queue.
            h = self.gen_polar_histogram(p_node)
            h = self.gen_bin_histogram(h)
            h = self.gen_masked_histogram(h, p_node, theta_node)
            
            discount = self.params.lmbda ** node_depth

            for theta_nxt in self.gen_directions(h, target_position - p_node):
                target_pos = p_node + self.params.D_t * self.theta_to_dxyz(theta_nxt)
                c_theta = self.dxyz_to_theta(target_pos - position)

                mu = (self.params.mu_1p, self.params.mu_2p, self.params.mu_3p) \
                        if node_depth + 1 != self.params.N_g else (self.params.mu_1, self.params.mu_2, self.params.mu_3)
                # todo refactor so no need to do awkward packing above...

                cost_node = cost + discount * (mu[0] * max(
                                                self.angle_dist(c_theta, theta_end),
                                                self.angle_dist(theta_nxt, theta_end)) +
                                               mu[1] * self.angle_dist(theta, c_theta) +
                                               mu[2] * self.angle_dist(c_theta, theta_node))
                sd = start_dir or theta_nxt
                heapq.heappush(visit_queue, (cost_node, (unique_ctr := unique_ctr + 1), (target_pos, theta_nxt), node_depth+1, sd))
        
        return None # fall through - should exit from loop.
        




        


if __name__ == '__main__':
    params = VFHParams( \
        D_t = 0.5,

        N_g = 5,
        R = 2,
        alpha = 36,
        b = 1, 
        mu_1 = 5,
        mu_2 = 2,
        mu_3 = 2,
        # (paper recommends mu_1 > mu_2 + mu_3)
        mu_1p = 5,
        mu_2p = 1,
        mu_3p = 1,
        # above parameters used to calculate cost for projected future motion rather than immediate plan
        lmbda = 0.8,
        hist_res = 0.5,

        r_active = 10, 
        r_drone = 0.5,

        t_low = 0.1,
        t_high = 0.9,
        
        mask_conf = 0.5,

        dir_spacing = 2
    )

    # simple test: sphere
    class SphereDummyHistogram:
        def __init__(self, pos, radius):
            self.pos = pos
            self.radius = radius
        def get_confidence(self, indices):
            indices2 = indices * params.hist_res
            if np.linalg.norm(indices2 - self.pos) <= self.radius:
                return 0.8
            return 0
    

    dh = SphereDummyHistogram(np.array([0,0,0]), 8)

    vfh = VFH2D(params, dh)
    
    

    pos = np.array([-10,0,0])

    phi = 0
    theta = 0 # + x direction
    t_pos = np.array([40,0,0])

    reslt = vfh.gen_polar_histogram(pos)
    np.savetxt('hist.dump', reslt, delimiter=',', newline='\n')

    

    reslt2 = vfh.gen_bin_histogram(reslt)
    np.savetxt('hist.dump2', reslt2, delimiter=',', newline='\n')
    reslt3 = vfh.gen_masked_histogram(reslt2,pos,theta)
    np.savetxt('hist.dump3', reslt3, delimiter=',', newline='\n')

    print(vfh.gen_directions(reslt3, t_pos - pos))
    reslt = vfh.get_target_dir(pos, theta,  t_pos)
    print(vfh.theta_to_dxyz(reslt))
