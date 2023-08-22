from .vfh_star import *

import open3d as o3d
import matplotlib.pyplot as plt


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
        def get_confidence(self, indices):
            indices2 = indices * params.hist_res
            query_point = o3d.core.Tensor([list(indices2)], dtype=o3d.core.Dtype.Float32)
            if self.scene.compute_signed_distance(query_point) < 0:
                return 0.8
            return 0    


    meshes = [sphere_mesh]
    dh = MeshTestHistogram(meshes)

    vfh = VFH(params, dh)

    display_scene = o3d.t.geometry.RaycastingScene()
    for mesh in meshes:
        display_scene.add_triangles(mesh)
    display_scene.add_triangles(o3d.t.geometry.TriangleMesh.create_sphere(radius=radius).translate(drone_pos))

    rays = o3d.t.geometry.RaycastingScene.create_rays_pinhole(
        fov_deg=90,
        center=[0, 0, 0],
        eye= drone_pos + [-1,1,0],
        up=[0, 1, 0],
        width_px=640,
        height_px=480,
    )
    # We can directly pass the rays tensor to the cast_rays function.
    ans = display_scene.cast_rays(rays)
    plt.imshow(ans['t_hit'].numpy())
    plt.show()
    

    pos = np.array(drone_pos)

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

    
    class ParallelWalls:
        def __init__(self, pos, width, height):
            self.pos = pos
            self.width = width
            self.height = height

        def get_confidence(self,indices, width, height):
            if (indices[0] == -self.width // 2 or indices[0] == self.width // 2) and -self.height // 2 < indices[2] < self.height // 2:
                return 0.7
            return 0
        
    
    pwh = ParallelWalls(np.array([0, 0, 0]), 20, 20)

    vfh = VFH(params, pwh)

    pos = np.array([-10,0,0])

    phi = 0
    theta = 0 # + x direction
    t_pos = np.array([40,0,0])

    reslt = vfh.gen_polar_histogram(pos)
    np.savetxt('hist_pw.dump', reslt, delimiter=',', newline='\n')


    reslt2 = vfh.gen_bin_histogram(reslt)
    np.savetxt('hist_pw.dump2', reslt2, delimiter=',', newline='\n')
    reslt3 = vfh.gen_masked_histogram(reslt2,pos,theta,phi)
    np.savetxt('hist_pw.dump3', reslt3, delimiter=',', newline='\n')

    print(vfh.gen_directions(reslt3, t_pos - pos))
    reslt = vfh.get_target_dir(pos, theta, phi,  t_pos)
    print(vfh.theta_phi_to_dxyz(*reslt))

