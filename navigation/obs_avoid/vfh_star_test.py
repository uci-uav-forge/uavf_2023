from .vfh_star import *
from .vfh_star_2d import *

import open3d as o3d
import matplotlib.pyplot as plt
import pprofile
import functools

default_params = VFHParams( \
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

def run_test(VFHImpl, histogram, drone_pos, phi, theta, t_pos, suffix):
    print("="*10)
    print(f"test{suffix}")
    vfh = VFHImpl(default_params, histogram)
    pos = np.array(drone_pos)

    reslt = vfh.gen_polar_histogram(pos)
    np.savetxt(f'polar_hist{suffix}', reslt, delimiter=',', newline='\n')

    

    reslt2 = vfh.gen_bin_histogram(reslt)
    np.savetxt(f'bin_hist{suffix}', reslt2, delimiter=',', newline='\n')
    reslt3 = vfh.gen_masked_histogram(reslt2,pos,theta,phi)
    np.savetxt(f'masked_hist{suffix}', reslt3, delimiter=',', newline='\n')

    print(vfh.gen_directions(reslt3, t_pos - pos))
    with pprofile.Profile() as prof:
        reslt = vfh.get_target_dir(pos, theta, phi,  t_pos)
        prof.dump_stats(f'profdump{suffix}')
    print(vfh.theta_phi_to_dxyz(*reslt))


class VFH2DWrapper(VFH2D):
    def get_masked_histogram(self, a,b,c, *args):
        super().get_masked_histogram(self, a,b,c)
    def get_target_dir(self, position: np.ndarray, theta: float, *args) -> tuple[float, float]:
        return [super().get_target_dir(position, theta, args[-1])]
    def gen_masked_histogram(self, bin_hist: np.ndarray, pos: np.array, theta: float, *args) -> np.ndarray:
        return super().gen_masked_histogram(bin_hist, pos, theta)
    def theta_phi_to_dxyz(self, theta):
        return super().theta_to_dxyz(theta)


if __name__ == '__main__':
    drone_pos = [-5,0,0]

    radius = 0.5
    sphere_mesh = o3d.t.geometry.TriangleMesh.create_sphere(radius=radius)

    # simple test: sphere
    class MeshTestHistogram:
        def __init__(self, meshes):
            self.meshes = meshes
            self.scene = o3d.t.geometry.RaycastingScene()
            for mesh in self.meshes:
                self.scene.add_triangles(mesh)
        @functools.lru_cache(maxsize=None)
        def confinner(self, indices):
            indices = np.array(indices)
            indices2 = indices * default_params.hist_res
            query_point = o3d.core.Tensor([list(indices2)], dtype=o3d.core.Dtype.Float32)
            if self.scene.compute_signed_distance(query_point) < 0:
                return 0.8
            return 0    

        def get_confidence(self, indices):
            return self.confinner(tuple(indices))

    meshes = [sphere_mesh]
    dh = MeshTestHistogram(meshes)

    display_scene = o3d.t.geometry.RaycastingScene()
    for mesh in meshes:
        display_scene.add_triangles(mesh)
    display_scene.add_triangles(o3d.t.geometry.TriangleMesh.create_sphere(radius=radius).translate(drone_pos))

    rays = o3d.t.geometry.RaycastingScene.create_rays_pinhole(
        fov_deg = 90,
        center=[0, 0, 0],
        eye= np.array(drone_pos) + np.array([-1,1,0]),
        up=[0, 1, 0],
        width_px=640,
        height_px=480,
    )
    # We can directly pass the rays tensor to the cast_rays function.
    ans = display_scene.cast_rays(rays)
    plt.imshow(ans['t_hit'].numpy())
    plt.show()

    phi = 0
    theta = 0 # + x direction
    t_pos = np.array([40,0,0])
    
    reslt = run_test(VFH2DWrapper, MeshTestHistogram(meshes), drone_pos, phi, theta, t_pos, '_open3d')

    
    class ParallelWalls:
        def __init__(self, pos, width, height):
            self.pos = pos
            self.width = width
            self.height = height

        def get_confidence(self,indices):
            if (indices[0] == -self.width // 2 or indices[0] == self.width // 2) and -self.height // 2 < indices[2] < self.height // 2:
                return 0.7
            return 0
        
    
    pwh = ParallelWalls(np.array([0, 0, 0]), 20, 20)

    pos = np.array([-10,0,0])

    phi = 0
    theta = 0 # + x direction
    t_pos = np.array([40,0,0])

    reslt = run_test(VFH, pwh, pos, phi, theta, t_pos, "_par_walls")
