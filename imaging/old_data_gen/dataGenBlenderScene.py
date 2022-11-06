import bpy
from dataGenRandom import *
from dataGenFileHandler import get_shapes, get_alphas
import uuid
import mathutils
from dataGenTarget import * #get_bbox, Target
from dataGenConstants import SHAPE_OFFSETS
def setup_blender_gpu():
    bpy.context.preferences.addons['cycles'].preferences.compute_device_type = 'CUDA'
    bpy.context.scene.cycles.device = 'GPU'
class BlenderScene:
    def __init__(self, resolution, focal, pitchBound = (0,0), rollBound = (0,0), altitudeBound = (50,100)):
        self.resolution = resolution
        self.focal = focal
        self.pitchBound = pitchBound
        self.rollBound = rollBound
        self.altitudeBound = altitudeBound

    def setupBlenderSettings(self):
        #reset blender environment
        bpy.ops.wm.read_factory_settings(use_empty=True)
        #setup resolution of the scene
        bpy.data.scenes['Scene'].render.resolution_x = self.resolution
        bpy.data.scenes['Scene'].render.resolution_y = self.resolution
        bpy.context.scene.render.image_settings.file_format = 'JPEG'

        #lighting settings
        #uncomment if data is too weird
        
        #sun location
        sun_loc = mathutils.Vector([randomNormal(0,1000), randomNormal(0,1000), 1500])
        #placing the sun at sun location
        bpy.ops.object.light_add(type='SUN', location=sun_loc)
        

    def setupCamera(self):
        #randomize camera position/orientation
        pitch, roll = randomOrientation(self.pitchBound, self.rollBound)
        altitude = randomAltitude(self.altitudeBound)

        #position camera in scene
        bpy.ops.object.camera_add(location=(0,0,altitude))



        #camera settings
        self.cam = bpy.context.object
        self.cam.data.clip_end = 1e08
        self.cam.data.type = 'PERSP'
        self.cam.data.lens_unit = 'MILLIMETERS'
        self.cam.data.lens = self.focal
        bpy.context.scene.camera = self.cam
        for obj in bpy.context.selected_objects:
            obj.select_set(False)
        for obj in bpy.context.scene.objects:
            if obj.type == 'CAMERA':
                obj.select_set(True)
        bpy.ops.transform.rotate(value=pitch, orient_axis='X')
        bpy.ops.transform.rotate(value=roll, orient_axis='Y')

        for obj in bpy.context.selected_objects:
            if obj.type == 'CAMERA':
                obj.select_set(False)
    def placeTargets(self, image_path, targetList, backgnd_list, shapeBound, alphaRatio):
        widthLower, widthUpper = shapeBound
        zOffset = 0
        targets = []
        print(targetList)
        for shape_name, shape_path, alpha_name, alpha_path, shape_i in targetList:
            shape_id = uuid.uuid4().hex
            alpha_id = uuid.uuid4().hex

            #import shape
            bpy.ops.import_scene.obj(filepath = shape_path)
            bpy.data.objects[shape_name].name = shape_id
            bpy.data.objects[shape_id].data.name = shape_id
            #import alphanumeric
            bpy.ops.import_scene.obj(filepath = alpha_path)
            bpy.data.objects[alpha_name].name = alpha_id
            bpy.data.objects[alpha_id].data.name = alpha_id

            #select imported objects and 
            bpy.ops.object.select_all(action='DESELECT')
            bpy.data.objects[shape_id].select_set(True)
            bpy.data.objects[alpha_id].select_set(True)
            bpy.ops.object.origin_set(type='ORIGIN_GEOMETRY', center='BOUNDS')
            dg = bpy.context.evaluated_depsgraph_get() #no idea what this is
            bpy.data.objects[shape_id].rotation_mode = 'XYZ'
            bpy.ops.object.select_all(action='DESELECT')

            #get random Color and give shape and alpha color
            scolor, acolor = randomizeColors()

            smat = bpy.data.materials.new('object_color')
            smat.diffuse_color = scolor

            amat = bpy.data.materials.new('alpha_color')
            amat.diffuse_color = acolor

            bpy.data.objects[shape_id].active_material = smat
            bpy.data.objects[alpha_id].active_material =  amat
            
            #give shape random size
            shapew = randomUniform(widthLower, widthUpper)
            scale = shapew / max(list(bpy.data.objects[shape_id].dimensions))
            bpy.data.objects[shape_id].select_set(True)
            bpy.ops.transform.resize(value=(scale,scale,scale), orient_type='LOCAL')
            bpy.data.objects[shape_id].select_set(False)
            #give alpha scaled down size of shape
            bpy.ops.object.select_all(action='DESELECT')
            alpha_scale = alphaRatio * max(list(bpy.data.objects[shape_id].dimensions))
            alpha_scale = alpha_scale / max(list(bpy.data.objects[alpha_id].dimensions))
            bpy.data.objects[shape_id].select_set(False)
            bpy.data.objects[alpha_id].select_set(True)
            bpy.ops.transform.resize(value=(alpha_scale, alpha_scale, alpha_scale), orient_type='LOCAL')


            letteroffset = mathutils.Vector(SHAPE_OFFSETS[shape_id]) * max(list(bpy.data.objects[shape_id].dimensions))
            #places object randomly within plane of camera view
            plane_co = (0,0,0)
            plane_no = (0,0,1)
            mw = self.cam.matrix_world
            o = mw.translation
            tr,br,bl,tl = [mw @ f for f in self.cam.data.view_frame(scene=bpy.context.scene)]

            x = tr - tl
            y = tr - br

            cx, cy = randomUniform(0.1,0.9), randomUniform(0.1,0.8)

            v = (bl + (cx * x + cy * y)) - o

            pt = mathutils.geometry.intersect_line_plane(o,o+v, plane_co, plane_no, True)

            #rotating letter randomly
            letter_rotation = randomRotation()
            bpy.data.objects[alpha_id].select_set(True)
            bpy.ops.transform.rotate(value=letter_rotation)
            bpy.data.objects[alpha_id].select_set(False)

            #place shape
            pt += mathutils.Vector([0,0,zOffset])
            zOffset += 5e-1
            bpy.data.objects[shape_id].location = pt
            bpy.data.objects[alpha_id].location = ( pt + mathutils.Vector([0,0,5e-2]) + letteroffset )

            #set shape as parent to letter
            bpy.data.objects[alpha_id].select_set(True)
            bpy.data.objects[shape_id].select_set(True)
            bpy.context.view_layer.objects.active = bpy.data.objects[shape_id]
            bpy.ops.object.parent_set()
            bpy.data.objects[alpha_id].select_set(False)
            bpy.ops.transform.rotate(value=letter_rotation, orient_axis='Z') #letter rotates w/ shape
            bpy.data.objects[shape_id].select_set(False)

            x,y,w,h = get_bbox(bpy.context.scene, bpy.context.scene.camera, bpy.data.objects[shape_id])

            #get bbox info for labels
            xCent, yCent = (x + (w/2)) / self.resolution, (y + (h/2)) / self.resolution
            normwidth, normheight = (w / self.resolution, h / self.resolution)
            targets.append(Target(xCent, yCent, normwidth, normheight, shape_name, alpha_name, letter_rotation, shape_i))

        #lay the background in the blender environment
        bgpath = randomBackground(backgnd_list)
        bpy.context.scene.render.film_transparent = True
        bpy.context.scene.use_nodes = True
        composite = bpy.context.scene.node_tree.nodes[0]
        render_layers = bpy.context.scene.node_tree.nodes[1]
        alpha_over = bpy.context.scene.node_tree.nodes.new(type='CompositorNodeAlphaOver')
        background_img_node = bpy.context.scene.node_tree.nodes.new(type='CompositorNodeImage')
        scale_node = bpy.context.scene.node_tree.nodes.new(type='CompositorNodeScale')
        #links?
        links = bpy.context.scene.node_tree.links
        link_1 = links.new(render_layers.outputs[0], alpha_over.inputs[2])
        link_2 = links.new(alpha_over.outputs[0], composite.inputs[0])
        link_3 = links.new(background_img_node.outputs[0], scale_node.inputs[0])
        link_4 = links.new(scale_node.outputs[0], alpha_over.inputs[1])
        bpy.data.scenes['Scene'].node_tree.nodes['Scale'].space = 'RENDER_SIZE'

        #load up background
        background_img_node.image = bpy.data.images.load(bgpath)
        bpy.data.scenes['Scene'].render.filepath = image_path
        bpy.ops.render.render(write_still=True)

        return targets
