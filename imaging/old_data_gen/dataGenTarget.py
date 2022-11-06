import bpy

def clamp(x, mini, maxi):
    return max(mini, min(x, maxi))

def get_bbox(scene, cam_ob, me_ob):
    mat = cam_ob.matrix_world.normalized().inverted()
    depsgraph = bpy.context.evaluated_depsgraph_get()
    mesh_eval = me_ob.evaluated_get(depsgraph)
    me = mesh_eval.to_mesh()
    me.transform(me_ob.matrix_world)
    me.transform(mat)
    camera = cam_ob.data
    frame = [-v for v in camera.view_frame(scene=scene)[:3]]
    camera_persp = camera.type != 'ORTHO'
    lx = []
    ly = []
    for v in me.vertices:
        co_local = v.co
        z = -co_local.z
        if camera_persp:
            if z == 0.0:
                lx.append(0.5)
                ly.append(0.5)
            else:
                frame = [(v / (v.z / z)) for v in frame]
        min_x, max_x = frame[1].x, frame[2].x
        min_y, max_y = frame[0].y, frame[1].y
        x = (co_local.x - min_x) / (max_x - min_x)
        y = (co_local.y - min_y) / (max_y - min_y)
        lx.append(x)
        ly.append(y)

    min_x = clamp(min(lx), 0.0, 1.0)
    max_x = clamp(max(lx), 0.0, 1.0)
    min_y = clamp(min(ly), 0.0, 1.0)
    max_y = clamp(max(ly), 0.0, 1.0)

    mesh_eval.to_mesh_clear()
    r = scene.render
    fac = r.resolution_percentage * 0.01
    dim_x = r.resolution_x * fac
    dim_y = r.resolution_y * fac
    if round((max_x - min_x) * dim_x) == 0 or round((max_y - min_y) * dim_y) == 0:
        return (0,0,0,0)
    return ( round(min_x * dim_x), round(dim_y - max_y * dim_y), #x,y
             round((max_x - min_x) * dim_x), #width
             round((max_y - min_y) * dim_y)) #height


class Target:
    def __init__(self, xCenter, yCenter, width, height, shape, alphaNum, orientation, id_):
        self.xCenter = xCenter
        self.yCenter = yCenter
        self.width = width
        self.height = height

        self.shape = shape
        self.alphaNum = alphaNum
        self.orientation = orientation
        self.id = id_
