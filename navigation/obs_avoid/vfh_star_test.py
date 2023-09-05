from .vfh_star import *
from .vfh_star_2d import *

import open3d as o3d
import matplotlib.pyplot as plt
import pprofile
import functools
import copy
import time

default_2d_params = VFHParams( \
        D_t = 0.2,

        N_g = 2,
        R = 2,
        alpha = 10,
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

class MeshTestHistogram:
    def __init__(self, meshes):
        self.meshes = meshes
        self.scene = o3d.t.geometry.RaycastingScene()
        for mesh in self.meshes:
            self.scene.add_triangles(mesh)
        # prefill cache - dont mess up profiling
        for x in range(-20,21):
            for y in range(-20,21):
                for z in range(-20,21):
                    self.confinner((x,y,z))
    @functools.cache
    def confinner(self, indices):
        indices = np.array(indices)
        indices2 = indices * default_2d_params.hist_res
        query_point = o3d.core.Tensor([list(indices2)], dtype=o3d.core.Dtype.Float32)
        if self.scene.compute_signed_distance(query_point) < 0:
            return 0.8
        return 0    

    def get_confidence(self, indices):
        return self.confinner(tuple(indices))


def run_test(VFHImpl, histogram, drone_pos, phi, theta, t_pos, suffix, do_profile=False):
    print("="*10)
    print(f"test{suffix}")
    vfh = VFHImpl(default_2d_params, histogram)
    pos = np.array(drone_pos)

    reslt = vfh.gen_polar_histogram(pos)
    np.savetxt(f'polar_hist{suffix}', reslt, delimiter=',', newline='\n')

    

    reslt2 = vfh.gen_bin_histogram(reslt)
    np.savetxt(f'bin_hist{suffix}', reslt2, delimiter=',', newline='\n')
    reslt3 = vfh.gen_masked_histogram(reslt2,pos,theta,phi)
    np.savetxt(f'masked_hist{suffix}', reslt3, delimiter=',', newline='\n')

    print("generated directions", vfh.gen_directions(reslt3, t_pos - pos))
    
    if do_profile:
        with pprofile.Profile() as prof:
            reslt = vfh.get_target_dir(pos, theta, phi,  t_pos)
        print(f"took {prof.total_time}")
        prof.dump_stats(f'profile{suffix}')
    else:
        t0 = time.time()
        reslt = vfh.get_target_dir(pos, theta, phi,  t_pos)
        t1 = time.time()
        print(f"took {t1-t0}")
    
    print("generated direction", vfh.theta_phi_to_dxyz(*reslt))
    return reslt

def run_mesh_test(VFHImpl, meshes, drone_pos, phi, theta, t_pos, suffix):
    dh = MeshTestHistogram(meshes)
    reslt = run_test(VFHImpl, dh, drone_pos, phi, theta, t_pos, suffix)

    display_scene = o3d.t.geometry.RaycastingScene()
    for mesh in meshes:
        display_scene.add_triangles(mesh)
    display_scene.add_triangles(o3d.t.geometry.TriangleMesh.create_sphere(radius=0.25).translate(drone_pos))
    if len(reslt) > 1:
        display_scene.add_triangles(o3d.t.geometry.TriangleMesh.create_arrow()
                                    .scale(0.1, (0,0,0))
                                    .rotate(o3d.geometry.get_rotation_matrix_from_xyz([0,math.pi/2,0]), center = (0,0,0))
                                    .rotate(o3d.geometry.get_rotation_matrix_from_xyz([0,reslt[1],reslt[0]]), center = (0,0,0))
                                    .translate(drone_pos))
    else:
        display_scene.add_triangles(o3d.t.geometry.TriangleMesh.create_arrow()
                                    .scale(0.1, (0,0,0))
                                    .rotate(o3d.geometry.get_rotation_matrix_from_xyz([0,math.pi/2,0]), center = (0,0,0))
                                    .rotate(o3d.geometry.get_rotation_matrix_from_xyz([0,0,reslt[0]]), center = (0,0,0))
                                    .translate(drone_pos))


    print("final result angle", *reslt)

    rays = o3d.t.geometry.RaycastingScene.create_rays_pinhole(
        fov_deg = 45,
        center= np.array([0,0,0]),
        eye= np.array(drone_pos) + np.array([-10,0,-10]), # kinda hardcoded...
        up=[0, 0, 1],
        width_px=640,
        height_px=480,
    )
    ans = display_scene.cast_rays(rays)
    plt.imshow(ans['t_hit'].numpy())
    plt.title(f'dir_below{suffix}')
    plt.savefig(f'dir_below{suffix}.png')
    plt.clf()
    
    rays = o3d.t.geometry.RaycastingScene.create_rays_pinhole(
        fov_deg = 45,
        center= np.array([0,0,0]),
        eye= np.array(drone_pos) + np.array([-10,0,10]), # kinda hardcoded...
        up=[0, 0, 1],
        width_px=640,
        height_px=480,
    )
    ans = display_scene.cast_rays(rays)
    plt.imshow(ans['t_hit'].numpy())
    plt.title(f'dir_above{suffix}')
    plt.savefig(f'dir_above{suffix}.png')
    plt.clf()

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
    drone_pos = [-3,0,0]

    radius = 0.7
    sphere_mesh = lambda: o3d.t.geometry.TriangleMesh.create_sphere(radius=radius)

    dead_end_meshes = [sphere_mesh(),
              sphere_mesh().translate([-1,1,0]),
              sphere_mesh().translate([-2,2,0]),
              sphere_mesh().translate([-1,-1,0]),
              sphere_mesh().translate([-2,-2,0])]

    phi = 0
    theta = 0 # + x direction
    t_pos = np.array([40,0,0])

    run_mesh_test(VFH2DWrapper, dead_end_meshes, drone_pos, phi, theta, t_pos, "_dead_end")
    run_mesh_test(VFH, dead_end_meshes, drone_pos, phi, theta, t_pos, "_dead_end_non_2d")

    turn_right_meshes = [sphere_mesh(),
              sphere_mesh().translate([-1,-1,0]),
              sphere_mesh().translate([-2,-2,0])]


    run_mesh_test(VFH2DWrapper, turn_right_meshes, drone_pos, phi, theta, t_pos, "_turn_right")

    parallel_wall_meshes = [sphere_mesh().translate([x,y,0]) for x in range(-10, 10) for y in (-5, 5)]

    run_mesh_test(VFH2DWrapper, parallel_wall_meshes, drone_pos, phi, theta, t_pos, "_par_walls")
    
    turn_left_meshes = [sphere_mesh()
                        .translate([x,y,0])
                        .rotate(o3d.geometry.get_rotation_matrix_from_xyz([0,0,-math.pi/4]), center = drone_pos) for x in range(-10, 10) for y in (-7, 7)]
    run_mesh_test(VFH2DWrapper, turn_left_meshes, drone_pos, phi, theta, t_pos, "_turn_left_par")