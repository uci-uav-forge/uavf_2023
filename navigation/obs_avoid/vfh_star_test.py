from .vfh_star import *


if __name__ == '__main__':
    params = VFHParams( \
        D_t = 0.5,

        N_g = 3,
        R = 2,
        alpha = 8,
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

    vfh = VFH(params, dh)
    
    

    pos = np.array([-10,0,0])

    phi = 0
    theta = 0 # + x direction
    t_pos = np.array([40,0,0])

    reslt = vfh.gen_polar_histogram(pos)
    np.savetxt('hist.dump', reslt, delimiter=',', newline='\n')

    

    reslt2 = vfh.gen_bin_histogram(reslt)
    np.savetxt('hist.dump2', reslt2, delimiter=',', newline='\n')
    reslt3 = vfh.gen_masked_histogram(reslt2,pos,theta,phi)
    np.savetxt('hist.dump3', reslt3, delimiter=',', newline='\n')

    print(vfh.gen_directions(reslt3, t_pos - pos))
    reslt = vfh.get_target_dir(pos, theta, phi,  t_pos)
    print(vfh.theta_phi_to_dxyz(*reslt))
