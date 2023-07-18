from dataclasses import dataclass
import numpy as np
import math

@dataclass
class VFHParams:
    D_t: float # total distance in meters planned per VFH iteration
    N_g: int # number of steps to plan out at a time
    R: float # steering radius: left/right turns
    R_y: float # steering radius: up/down turns
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
    r_drone: float # distance to consider our width as for widening obstacles in polar histogram

    t_low: float # low threshold for binary histogram
    t_high: float # high threshold


class VFH:
    def __init__(self, hist, params: VFHParams):
        self.hist = hist
        self.params = params

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
                    mx = dx*self.params.hist_res
                    my = dy*self.params.hist_res
                    mz = dz*self.params.hist_res
                    dist = (mx**2 + my**2 + mz**2)**0.5
                    dxy = (dx**2 + dy**2)**0.5
                    if self.params.r_drone <= dist <= self.params.r_active:
                        idx2 = idx + np.array([dx,dy,dz])
                        conf = self.hist.get_confidence(idx2)

                        theta = math.atan2(dy,dx)  # k * alpha

                        phi = math.atan(dz/dxy) + math.pi/2 if dxy != 0 else math.pi*abs(dz)/dz
                        

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
                    results[i][j] = results[(i-1) % self.params.alpha][(j-1) % 2 * self.params.alpha]
        
        return results

    def gen_masked_histogram(self, bin_polar_hist: np.ndarray, theta: float) -> np.ndarray:
        dx_r = self.params.R * math.cos(theta)
        dy_r = self.params.R * math.sin(theta)
        dx_l = -self.params.R * math.cos(theta)
        dx_r = -self.params.R * math.sin(theta)
        # TODO finish implementation



if __name__ == '__main__':
    params = VFHParams( \
        D_t = 2,

        N_g = 5,
        R = 2,
        R_y = 2,
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
        t_high = 0.9

    )

    # simple test: giant wall
    class GiantWallDummyHistogram:
        def get_confidence(self,indices):
            if indices[0] > 0:
                return 0.7
            return 0
    

    vfh = VFH(GiantWallDummyHistogram(), params)

    reslt = vfh.gen_polar_histogram(np.array([-5,0,0]))
    np.savetxt('hist.dump', reslt, delimiter=',', newline='\n')