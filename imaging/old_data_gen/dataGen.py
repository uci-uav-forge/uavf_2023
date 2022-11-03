import bpy

'''

    #########################################
    ######### README ########################
    #########################################
    Program contains an Argument Parser.
        --Details of it are in dataGenConstants.py
    Contents of dataGenConstants.py
        --SHAPE_SCALES defaultdict
        --SHAPE_OFFSETS defaultdict
        --OPTS parsed arguments with details attributes defined in file
'''
import sys
import os
print(os.getcwd())
sys.path.append(os.getcwd())
#include constants

#this has all of the constants INCLUDING the Argument PARSER
import dataGenConstants
import dataGenFileHandler
import dataGenBlenderScene
import dataGenRandom
#dataGen stuff
OPTS = dataGenConstants.OPTS

if __name__ == '__main__':
    #make necessary folders
    dataGenFileHandler.format_directory(OPTS.dir)

    dataGenBlenderScene.setup_blender_gpu()
    scene = dataGenBlenderScene.BlenderScene(OPTS.resolution, OPTS.focal_len,
                                        pitchBound=(OPTS.lower_pitch, OPTS.upper_pitch),
                                        rollBound=(OPTS.lower_roll, OPTS.upper_roll),
                                        altitudeBound=(OPTS.lower_altitude, OPTS.upper_altitude))


    shape_list = dataGenFileHandler.get_shapes(OPTS.shape_dir)
    alpha_list = dataGenFileHandler.get_alphas(OPTS.alpha_dir)
    bkg_list = dataGenFileHandler.get_backgrounds(OPTS.backgnd)
    for idxData in range(OPTS.n):
        scene.setupBlenderSettings()
        scene.setupCamera()

        image_dir = f'{OPTS.dir}/images/{idxData}.jpg'
        print(image_dir)
        shape_list = dataGenFileHandler.get_shapes(OPTS.shape_dir)
        print("SHAPE DIR", OPTS.shape_dir)
        alpha_list = dataGenFileHandler.get_alphas(OPTS.alpha_dir)
        target_list = dataGenRandom.randomGenerateTargets(shape_list, alpha_list, OPTS.shape_dir, OPTS.alpha_dir)
        targets = scene.placeTargets(image_dir, target_list, bkg_list, (OPTS.min_width, OPTS.max_width), OPTS.alphascale)

        with open(f'{OPTS.dir}/labels/{idxData}.txt','w') as file:
            for target in targets:
                file.write(f'{target.id} {target.xCenter} {target.yCenter} {target.width} {target.height}\n')

        with open(f'{OPTS.dir}/full_labels/{idxData}.txt','w') as file:
            for target in targets:
                file.write(f'{target.shape} {target.xCenter} {target.yCenter} {target.width} {target.height} {target.alphaNum} {target.orientation}\n')
        
    print('Finished') 
