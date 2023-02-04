# test the obstacle avoidance code visually using matplotlib
# requires obstacle_avoidance.py to be in the same folder
# obstacles can be either defined in the code, or input through the terminal
# change to whatever you prefer in the test() function

from obstacle_avoidance import *
import matplotlib.pyplot as plt
import numpy as np
import time

def test():
    '''comment out either [1] or [2]'''
    # [1] below obstacles are manually defined in code
    o1 = obstacle(pos=(5,8), rad=0.7)
    o2 = obstacle(pos=(-3, 8), rad=1)
    o3 = obstacle(pos=(2.5, 3.5), rad=0.5)
    obstacles = [o1, o2, o3]

    # [2] below input() is used to ask user in terminal for obstacles
    #obstacles = get_obstacles()

    # find heading
    print('\n calculating heading...')
    t1 = time.perf_counter()
    heading = avoidObstacle(objects=obstacles)
    t2 = time.perf_counter()
    print(f'The heading is {heading}. It took {t2 - t1} seconds to find')

    # plot obstacles and headign:
    print('\nplotting...')
    fig, ax = create_plot()
    add_obstacles_to_plot(ax, obstacles)
    add_heading_to_plot(plt, ax, heading)

    plt.legend(loc="lower right")
    plt.show()


def get_obstacles() -> list[obstacle]:
    '''ask user to input '''
    obstacles = []

    while True:
        x = int(input("Enter x-coordinate of obstacle: "))
        y = int(input("Enter y-coordinate of obstacle: "))
        r = int(input("Enter radius of obstacle: "))
        obstacles.append(obstacle(pos=(x,y), rad=r))

        if not more_obstacles():
            break
    
    return obstacles

def more_obstacles() -> bool:
    '''asks user if htey want to add another obstacle, returnes true if they do'''
    x = input("Add another obstacle? [yes] or [no]: ")
    return x.lower() == 'yes' or x.lower() == 'y'

def add_obstacles_to_plot(ax, obstacles: list[obstacle]) -> None:
    '''plots obstacles on matplotlib'''
    
    ax.set_aspect(1)

    for o in obstacles:
        ax.add_artist(plt.Circle((o.get_position()[0],o.get_position()[1]), o.get_radius()))

def add_heading_to_plot(plt, ax, heading: int or float) -> None:
    '''plots the heading'''
    if heading == 0:
        plt.axvline(x = 0, color = 'g', label='suggested heading')
    else:
        x = np.linspace(-20, 20, 100)
        slope = math.tan(math.radians(heading))
        y = slope*x
        ax.plot(x,y,color='g', label='suggested heading')


def create_plot():
    '''creates matplotlib with correct boundaries and camera vision represented by lines'''
    fig, ax = plt.subplots()
    ax.set(xlim=(-20,20), ylim=(0,20)) # define boundaries. camera can see 6 meters (20 feet away)

    # plot camera view boundaries
    x = np.linspace(0, 20, 100)
    slope = math.tan(math.radians(43.5))
    y = x*slope # corresponds to 43.5 degrees
    ax.plot(x,y, color='r', label='maximum camera range')

    x = np.linspace(-20, 0, 100)
    slope = -1*slope
    y = x*slope
    ax.plot(x,y,'r')

    plt.title('Obstacle Avoidance Visualization')
    plt.xlabel('horizontal distance [feet]')
    plt.ylabel('vertical distance [feet]')

    return fig, ax


if __name__ == '__main__':
    test()