import numpy as np
from matplotlib import cm
import random
def poissonGen(x):
    num = np.random.poisson(lam=x)
    if num == 0: return 1
    else: return num

def randomizeColors():
    #return [Red,Green,Blue,Alpha]
    #alpha is brightness
    #all values are from 0 -> 1
    return ( np.random.uniform(), np.random.uniform(), np.random.uniform(), np.random.uniform() )

def randomizeColors(n_colors=1000):
    x = np.linspace(0,1,num=n_colors)
    colors = cm.get_cmap('rainbow')(x)
    clist = [(*tuple(c),1) for c in colors[:,:3]]
    letter_idx = np.random.choice(len(clist))
    shape_idx = (np.random.randint(len(clist) / 4, 3 * len(clist) / 4) + letter_idx) % len(clist)
    return clist[shape_idx], clist[letter_idx]

def randomOrientation(pitchBound: (float,float), rollBound: (float,float) ):
    pl, pu = pitchBound
    rl, ru = rollBound
    pitch = np.random.uniform(pl, pu) * np.pi / 180
    roll = np.random.uniform(rl, ru) * np.pi / 180
    return pitch,roll

def randomRotation():
    return np.random.uniform(0,2*np.pi)
def randomAltitude(altitudeBound):
    al, au = altitudeBound
    return np.random.uniform(al, au)

def randomNormal(mu, var):
    return np.random.normal(mu,var)

def randomUniform(lower,upper):
    return np.random.uniform(lower, upper)

def randomGenerateTargets(shape_list, alpha_list, shape_dir, alpha_dir):
    enum_shape = list(enumerate(shape_list))
    enum_alpha = list(enumerate(alpha_list))

    numberObjects = poissonGen(1.4)
    listTargets = []
    for _ in range(numberObjects):
        alpha_i, alpha_name = random.choice(enum_alpha)
        shape_i, shape_name = random.choice(enum_shape)

        shape_path = shape_dir + shape_name + '.obj'
        print("SHAPE PATH", shape_path)
        alpha_path = alpha_dir + alpha_name + '.obj'
        listTargets.append( (shape_name, shape_path, alpha_name, alpha_path, shape_i) )
    return listTargets


def randomBackground(backgnd_list):
    return random.choice(backgnd_list)
