import matplotlib.pyplot as plt  # plot, show
import math                      # atan2

points = [(985, 268), (112, 316), (998, 448), (1018, 453), (1279, 577), (1196, 477), (1161, 443), (986, 0), (830, 0), (983, 230), (998, 425), (998, 255)]

def sort_counterclockwise(points, centre = None):
    if centre:
        centre_x, centre_y = centre
    else:
        centre_x, centre_y = sum([x for x,_ in points])/len(points), sum([y for _,y in points])/len(points)
        angles = [math.atan2(y - centre_y, x - centre_x) for x,y in points]
        counterclockwise_indices = sorted(range(len(points)), key=lambda i: angles[i])
        counterclockwise_points = [points[i] for i in counterclockwise_indices]
    return counterclockwise_points

points1 = sort_counterclockwise(points)
print(points)
print(points1)