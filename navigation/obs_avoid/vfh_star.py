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


class VFH:
    def __init__(self, params: VFHParams, hist: 'Histogram'):
        self.params = params
        self.hist = hist

    def pos_to_idx(self, pos: np.array) -> np.array:
        return np.round(pos/self.params.hist_res)
    
    def gen_polar_histogram(self, pos: np.array) -> np.ndarray:
        result = np.zeros((self.params.alpha, 2*self.params.alpha))

        a = self.params.b * (self.params.r_active)**2
        # note: paper wants above to be a = 1 + b(r_active -1)**2/4
        # this gives weird negative results for far things ... idk what it's doing.

        idx = self.pos_to_idx(pos)
        di = math.ceil(self.params.r_active / self.params.hist_res)
        for dx in range(-di,di+1):
            for dy in range(-di, di+1):
                for dz in range(-di, di+1):
                    mx = (dx+0.5)*self.params.hist_res
                    my = (dy+0.5)*self.params.hist_res
                    mz = (dz+0.5)*self.params.hist_res
                    dist = (mx**2 + my**2 + mz**2)**0.5
                    dxy = (dx**2 + dy**2)**0.5
                    if self.params.r_drone <= dist <= self.params.r_active:
                        idx2 = idx + np.array([dx,dy,dz])
                        conf = self.hist.get_confidence(idx2)

                        theta = math.atan2(dy,dx)

                        phi = math.atan2(dz, dxy) + math.pi/2  

                        enlargment_angle = math.asin(self.params.r_drone/dist)

                        theta_idx = round((theta/(2*math.pi))*2*self.params.alpha)
                        phi_idx = round(phi/math.pi*self.params.alpha)

                        

                        enlarge_idx = math.ceil(enlargment_angle/(math.pi)*self.params.alpha)

                        for tidx in range(theta_idx - enlarge_idx, theta_idx + enlarge_idx+1):
                            for pidx in range(phi_idx - enlarge_idx, phi_idx + enlarge_idx+1):
                                result[pidx % self.params.alpha,tidx % (2*self.params.alpha)] += conf*conf*(a - self.params.b*dist*dist)
        return result
    
    def gen_bin_histogram(self, polar_hist: np.ndarray) -> np.ndarray:
        results = np.zeros((self.params.alpha, 2 * self.params.alpha))
        for i in range(self.params.alpha):
            for j in range(self.params.alpha * 2):
                if polar_hist[i][j] > self.params.t_high:
                    results[i][j] = 1
                elif polar_hist[i][j] < self.params.t_low:
                    results[i][j] = 0
                else:
                    # for now 
                    results[i][j] = 1 
        return results


    def gen_masked_histogram(self, bin_hist: np.ndarray, pos: np.array, theta: float, phi: float) -> np.ndarray:
        results = bin_hist.copy()

        # left and right displacements to mask out for each phi
        thetas = [[0, 0] for _ in range(self.params.alpha)] #todo change to np

        lx = pos[0] + self.params.R*math.cos(theta)*math.cos(phi)
        ly = pos[1] + self.params.R*math.sin(theta)*math.cos(phi)
        
        rx = pos[0] - self.params.R*math.cos(theta)*math.cos(phi)
        ry = pos[1] - self.params.R*math.sin(theta)*math.cos(phi)

        idx = self.pos_to_idx(pos)
        di = math.ceil(self.params.r_active / self.params.hist_res)
        # minor todo: iterator method for this pattern.
        for dx in range(-di,di+1):
            for dy in range(-di, di+1):
                for dz in range(-di, di+1):
                    mx = (dx+0.5)*self.params.hist_res
                    my = (dy+0.5)*self.params.hist_res
                    mz = (dz+0.5)*self.params.hist_res
                    dist = (mx**2 + my**2 + mz**2)**0.5
                    if not (self.params.r_drone <= dist <= self.params.r_active):
                        continue
                    # for now just run the old algorithm on each z-layer
                    # essentially instead of having a circle due to our turn radius we're using cylinders instead
                    # should be good enough.
                    #mz = (dz+0.5)*self.params.hist_res
                    
                    theta_pos = math.atan2(dy,dx)
                    dxy = (dx**2 + dy**2)**0.5
                    phi_pos = math.atan2(dz, dxy) + math.pi/2
                    phi_pos_idx = round(phi_pos/math.pi*self.params.alpha)

                    idx2 = idx + np.array([dx,dy,dz])

                    if self.hist.get_confidence(idx2) > self.params.mask_conf:

                        if ((mx-lx)**2+(my-ly)**2)**0.5 < (self.params.R + self.params.r_drone):
                            dist = ((theta + math.pi) - theta_pos) % 2*math.pi
                            if dist < math.pi:
                                thetas[phi_pos_idx][0] = max(thetas[phi_pos_idx][0], dist)

                        
                        if ((mx-rx)**2+(my-ry)**2)**0.5 < (self.params.R + self.params.r_drone):
                            dist = (theta_pos - (theta + math.pi)) % 2*math.pi 
                            if dist < math.pi:
                                thetas[phi_pos_idx][1] = max(thetas[phi_pos_idx][1], dist)
        
        for i in range(self.params.alpha):
            lt,rt = thetas[i]
            for j in range(self.params.alpha * 2):
                theta_here = j * 2*math.pi / self.params.alpha

                # check if it's in the range [-theta - left_theta, -theta + right_theta]
                
                dist = (theta_here - (theta+math.pi - lt)) % 2*math.pi

                if dist + 0.01 <= lt + rt:
                    # block it out
                    results[i][j] = 1
        return results


    def ij2step(self, i: int, j:int):
        ia = i/self.params.alpha *2* math.pi
        ja = j/(2*self.params.alpha) * 2*math.pi
        step_dist = self.params.D_t / self.params.N_g
        return np.array([step_dist*math.cos(ja)*math.cos(ia), step_dist*math.sin(ja)*math.cos(ia), step_dist*math.sin(ia)])
    

    def gen_directions(self, masked_hist: np.ndarray, delta_position: np.ndarray):
        theta_dpos = math.atan2(delta_position[1],delta_position[0])
        dxy = (delta_position[0]**2 + delta_position[1]**2)**0.5
        phi_dpos = math.atan2(delta_position[2], dxy) + math.pi/2

        dpos_j = round(theta_dpos * 2*self.params.alpha / (2*math.pi))
        dpos_i = round(phi_dpos * self.params.alpha / math.pi )

        # idea: sample in the target direction and spaced-out samples on borders of areas
        result = []
        too_close = [[False] * (2*self.params.alpha) for _ in range(self.params.alpha)]
        
        
        for i in range(self.params.alpha):
            for j in range(self.params.alpha * 2):
                if not masked_hist[i][j] and not too_close[i][j]:
                    aijs = [(i+1,j),(i-1,j),(i,j+1),(i,j-1)]
                    if any(masked_hist[ai % self.params.alpha][aj % (2*self.params.alpha)] for ai,aj in aijs):
                        result.append((i,j))
                        for di in range(-self.params.dir_spacing,self.params.dir_spacing+1):
                            for dj in range(-self.params.dir_spacing,self.params.dir_spacing+1):
                                too_close[(i+di)%self.params.alpha][(j+dj) % (2*self.params.alpha)] = True
        
        if not masked_hist[dpos_i][dpos_j] and not too_close[dpos_i][dpos_j]:
            result.append((dpos_i, dpos_j))
        
        return [ (math.pi / self.params.alpha * thi, math.pi / self.params.alpha * phi - math.pi/2) for phi, thi in result]

    def angle_dist(self, tp1, tp2):
        ad_inner = lambda a1, a2: max(abs(a1 - a2), abs(a1 - a2 + 2*math.pi), abs(a1 - a2 - 2*math.pi))
        return ad_inner(tp1[0], tp2[0]) + 10 * ad_inner(tp1[1], tp2[1])

    def theta_phi_to_dxyz(self, theta, phi):
        return np.array([math.cos(phi)*math.cos(theta), math.cos(phi)*math.sin(theta), math.sin(phi)])
    
    def dxyz_to_theta_phi(self, dxyz):
        theta = math.atan2(dxyz[1],dxyz[0])
        dxy = (dxyz[0]**2 + dxyz[1]**2)**0.5
        phi = math.atan2(dxyz[2], dxy)
        return (theta, phi)


    def get_target_dir(self, position: np.ndarray, theta: float, phi: float, target_position: np.ndarray) -> tuple[float, float]:
        # execute A* alrgorithm.
        # idea: repeatedly expand nodes based on heuristic until we reach a node that has been expanded N_g times.
        # the first step on the way to that node is the target.

        # queue: tuples of cost, state, depth

        visit_queue = [(0, None, (position, theta, phi), 0, None)]

        theta_end, phi_end = self.dxyz_to_theta_phi(target_position - position)

        unique_ctr = 0
        
        while len(visit_queue):
            cost, _, node_info, node_depth, start_dir = heapq.heappop(visit_queue)
            p_node, theta_node, phi_node = node_info
            if node_depth == self.params.N_g:
                return start_dir
            
            # otherwise expand node and insert into queue.
            h = self.gen_polar_histogram(p_node)
            h = self.gen_bin_histogram(h)
            h = self.gen_masked_histogram(h, p_node, theta_node, phi_node)
            
            discount = self.params.lmbda ** node_depth

            for theta_nxt, phi_nxt in self.gen_directions(h, target_position - p_node):
                target_pos = p_node + self.params.D_t * self.theta_phi_to_dxyz(theta_nxt, phi_nxt)
                c_theta, c_phi = self.dxyz_to_theta_phi(target_pos - position)

                mu = (self.params.mu_1p, self.params.mu_2p, self.params.mu_3p) \
                        if node_depth + 1 != self.params.N_g else (self.params.mu_1, self.params.mu_2, self.params.mu_3)
                # todo refactor so no need to do awkward packing above...

                cost_node = cost + discount * (mu[0] * max(
                                                self.angle_dist((c_theta, c_phi), (theta_end, phi_end)),
                                                self.angle_dist((theta_nxt, phi_nxt), (theta_end, phi_end))) +
                                               mu[1] * self.angle_dist((theta, phi), (c_theta, c_phi)) +
                                               mu[2] * self.angle_dist((c_theta, c_phi), (theta_node, phi_node)))
                sd = start_dir or (theta_nxt, phi_nxt)
                heapq.heappush(visit_queue, (cost_node, (unique_ctr := unique_ctr + 1), (target_pos, theta_nxt, phi_nxt), node_depth+1, sd))
        
        return None # fall through - should exit from loop.
        




        


if __name__ == '__main__':
    params = VFHParams( \
        D_t = 0.5,

        N_g = 5,
        R = 2,
        alpha = 20,
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

        dir_spacing = 20

    )

    # simple test: giant wall
    class GiantWallDummyHistogram:
        def get_confidence(self,indices):
            if indices[0] > 0:
                return 0.7
            return 0
    

    vfh = VFH(params, GiantWallDummyHistogram())

    pos = np.array([-5,0,0])
    phi = 0
    theta = 0 # + x direction
    t_pos = np.array([-3,5,0])

    reslt = vfh.gen_polar_histogram(pos)
    np.savetxt('hist.dump', reslt, delimiter=',', newline='\n')

    

    reslt2 = vfh.gen_bin_histogram(reslt)
    np.savetxt('hist.dump2', reslt2, delimiter=',', newline='\n')
    reslt3 = vfh.gen_masked_histogram(reslt2,pos,theta,phi)
    np.savetxt('hist.dump3', reslt3, delimiter=',', newline='\n')

    print(vfh.gen_directions(reslt3, t_pos))
    reslt = vfh.get_target_dir(pos, theta, phi,  t_pos)
    print(vfh.theta_phi_to_dxyz(*reslt))
