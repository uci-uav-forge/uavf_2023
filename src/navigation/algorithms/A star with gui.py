from random import randrange
from tkinter import *
import tkinter as tk
import math

class node:
    def __init__(self, number, x, y, g, h, parent):
        self.number = number
        self.x = x
        self.y = y
        self.g = g
        self.h = h
        self.f = g + h
        self.parent = parent

def calculateH(x, y):
    h = 0
    while x != goalX or y != goalY:
        if x == goalX and y < goalY:
            y = y + 1
            h = h + 1
        if x == goalX and y > goalY:
            y = y - 1
            h = h + 1
        if x > goalX and y == goalY:
            x = x - 1
            h = h + 1
        if x < goalX and y == goalY:
            x = x + 1
            h = h + 1
        if x < goalX and y < goalY:
            x = x + 1
            y = y + 1
            h = h + 1.4
        if x > goalX and y > goalY:
            x = x - 1
            y = y - 1
            h = h + 1.4
        if x > goalX and y < goalY:
            x = x - 1
            y = y + 1
            h = h + 1.4
        if x < goalX and y > goalY:
            x = x + 1
            y = y - 1
            h = h + 1.4
    return h

def checkNeighbors(parentNode):
    global counter
    # print("parentNode G: " + str(parentNode.g))
    print(coordinates)
    print("Parent: " + str(parentNode.x) + ", " + str(parentNode.y))
    if (parentNode.x + 1, parentNode.y) in coordinates:
        print("Changing G value")
        for ex in nodes:
            if ex.x == parentNode.x + 1 and ex.y == parentNode.y:
                if ex.g > parentNode.g + 1:
                    ex.g = parentNode.g + 1
                    ex.f = ex.g + calculateH(parentNode.x + 1, parentNode.y)
                    ex.parent = parentNode.number
                    nodes.sort(key=lambda x: x.f, reverse=True)
                break

    elif (parentNode.x + 1, parentNode.y) not in explored and (parentNode.x + 1,
                                                               parentNode.y) not in obstacles and parentNode.x + 1 > 0 and parentNode.x + 1 <= gridX and parentNode.y > 0 and parentNode.y <= gridY:
        nodes.append(
            node(counter, parentNode.x + 1, parentNode.y, parentNode.g + 1,
                 calculateH(parentNode.x + 1, parentNode.y),
                 parentNode.number))
        coordinates.append((parentNode.x + 1, parentNode.y))
        counter += 1
        print("adding " + str(nodes[len(nodes) - 1].x) + ", " + str(nodes[len(nodes) - 1].y))
        # print("adding " + str(nodes[len(nodes) - 1].x) + ", " + str(nodes[len(nodes) - 1].y) + " g: " + str(nodes[len(nodes) - 1].g) + " h: " + str(nodes[len(nodes) - 1].h) + " f: " + str(nodes[len(nodes) - 1].f))
        nodes.sort(key=lambda x: x.f, reverse=True)

    if (parentNode.x + 1, parentNode.y + 1) in coordinates:
        print("Changing G value")
        for ex in nodes:
            if ex.x == parentNode.x + 1 and ex.y == parentNode.y + 1:
                if ex.g > parentNode.g + 1:
                    ex.g = parentNode.g + 1
                    ex.f = ex.g + calculateH(parentNode.x + 1, parentNode.y + 1)
                    ex.parent = parentNode.number
                    nodes.sort(key=lambda x: x.f, reverse=True)
                break

    elif (parentNode.x + 1, parentNode.y + 1) not in explored and (parentNode.x + 1,
                                                                   parentNode.y + 1) not in obstacles and parentNode.x + 1 > 0 and parentNode.x + 1 <= gridX and parentNode.y + 1 > 0 and parentNode.y + 1 <= gridY:
        nodes.append(node(counter, parentNode.x + 1, parentNode.y + 1, parentNode.g + 1.4,
                          calculateH(parentNode.x + 1, parentNode.y + 1), parentNode.number))
        coordinates.append((parentNode.x + 1, parentNode.y + 1))
        counter += 1
        print("adding " + str(nodes[len(nodes) - 1].x) + ", " + str(nodes[len(nodes) - 1].y))
        # print("adding " + str(nodes[len(nodes) - 1].x) + ", " + str(nodes[len(nodes) - 1].y) + " g: " + str(nodes[len(nodes) - 1].g) + " h: " + str(nodes[len(nodes) - 1].h) + " f: " + str(nodes[len(nodes) - 1].f))
        nodes.sort(key=lambda x: x.f, reverse=True)

    if (parentNode.x, parentNode.y + 1) in coordinates:
        print("Changing G value")
        for ex in nodes:
            if ex.x == parentNode.x and ex.y == parentNode.y + 1:
                if ex.g > parentNode.g + 1:
                    ex.g = parentNode.g + 1
                    ex.f = ex.g + calculateH(parentNode.x, parentNode.y + 1)
                    ex.parent = parentNode.number
                    nodes.sort(key=lambda x: x.f, reverse=True)
                break

    elif (parentNode.x, parentNode.y + 1) not in explored and (parentNode.x,
                                                               parentNode.y + 1) not in obstacles and parentNode.x > 0 and parentNode.x <= gridX and parentNode.y + 1 > 0 and parentNode.y + 1 <= gridY:
        nodes.append(
            node(counter, parentNode.x, parentNode.y + 1, parentNode.g + 1,
                 calculateH(parentNode.x, parentNode.y + 1),
                 parentNode.number))
        coordinates.append((parentNode.x, parentNode.y + 1))
        counter += 1
        print("adding " + str(nodes[len(nodes) - 1].x) + ", " + str(nodes[len(nodes) - 1].y))
        # print("adding " + str(nodes[len(nodes) - 1].x) + ", " + str(nodes[len(nodes) - 1].y) + " g: " + str(nodes[len(nodes) - 1].g) + " h: " + str(nodes[len(nodes) - 1].h) + " f: " + str(nodes[len(nodes) - 1].f))
        nodes.sort(key=lambda x: x.f, reverse=True)

    if (parentNode.x - 1, parentNode.y + 1) in coordinates:
        print("Changing G value")
        for ex in nodes:
            if ex.x == parentNode.x - 1 and ex.y == parentNode.y + 1:
                if ex.g > parentNode.g + 1:
                    ex.g = parentNode.g + 1
                    ex.f = ex.g + calculateH(parentNode.x - 1, parentNode.y + 1)
                    ex.parent = parentNode.number
                    nodes.sort(key=lambda x: x.f, reverse=True)
                break

    elif (parentNode.x - 1, parentNode.y + 1) not in explored and (parentNode.x - 1,
                                                                   parentNode.y + 1) not in obstacles and parentNode.x - 1 > 0 and parentNode.x - 1 <= gridX and parentNode.y + 1 > 0 and parentNode.y + 1 <= gridY:
        nodes.append(node(counter, parentNode.x - 1, parentNode.y + 1, parentNode.g + 1.4,
                          calculateH(parentNode.x - 1, parentNode.y + 1), parentNode.number))
        coordinates.append((parentNode.x - 1, parentNode.y + 1))
        counter += 1
        print("adding " + str(nodes[len(nodes) - 1].x) + ", " + str(nodes[len(nodes) - 1].y))
        # print("adding " + str(nodes[len(nodes) - 1].x) + ", " + str(nodes[len(nodes) - 1].y) + " g: " + str(nodes[len(nodes) - 1].g) + " h: " + str(nodes[len(nodes) - 1].h) + " f: " + str(nodes[len(nodes) - 1].f))
        nodes.sort(key=lambda x: x.f, reverse=True)

    if (parentNode.x - 1, parentNode.y) in coordinates:
        print("Changing G value")
        for ex in nodes:
            if ex.x == parentNode.x - 1 and ex.y == parentNode.y:
                if ex.g > parentNode.g + 1:
                    ex.g = parentNode.g + 1
                    ex.f = ex.g + calculateH(parentNode.x - 1, parentNode.y)
                    ex.parent = parentNode.number
                    nodes.sort(key=lambda x: x.f, reverse=True)
                break

    if (parentNode.x - 1, parentNode.y) not in coordinates and (
            parentNode.x - 1, parentNode.y) not in explored and (parentNode.x - 1,
                                                                 parentNode.y) not in obstacles and parentNode.x - 1 > 0 and parentNode.x - 1 <= gridX and parentNode.y > 0 and parentNode.y <= gridY:
        nodes.append(
            node(counter, parentNode.x - 1, parentNode.y, parentNode.g + 1,
                 calculateH(parentNode.x - 1, parentNode.y),
                 parentNode.number))
        coordinates.append((parentNode.x - 1, parentNode.y))
        counter += 1
        print("adding " + str(nodes[len(nodes) - 1].x) + ", " + str(nodes[len(nodes) - 1].y))
        # print("adding " + str(nodes[len(nodes) - 1].x) + ", " + str(nodes[len(nodes) - 1].y) + " g: " + str(nodes[len(nodes) - 1].g) + " h: " + str(nodes[len(nodes) - 1].h) + " f: " + str(nodes[len(nodes) - 1].f))
        nodes.sort(key=lambda x: x.f, reverse=True)

    if (parentNode.x - 1, parentNode.y - 1) in coordinates:
        print("Changing G value")
        for ex in nodes:
            if ex.x == parentNode.x - 1 and ex.y == parentNode.y - 1:
                if ex.g > parentNode.g + 1:
                    ex.g = parentNode.g + 1
                    ex.f = ex.g + calculateH(parentNode.x - 1, parentNode.y - 1)
                    ex.parent = parentNode.number
                    nodes.sort(key=lambda x: x.f, reverse=True)
                break

    elif (parentNode.x - 1, parentNode.y - 1) not in explored and (parentNode.x - 1,
                                                                   parentNode.y - 1) not in obstacles and parentNode.x - 1 > 0 and parentNode.x - 1 <= gridX and parentNode.y - 1 > 0 and parentNode.y - 1 <= gridY:
        nodes.append(node(counter, parentNode.x - 1, parentNode.y - 1, parentNode.g + 1.4,
                          calculateH(parentNode.x - 1, parentNode.y - 1), parentNode.number))
        coordinates.append((parentNode.x - 1, parentNode.y - 1))
        counter += 1
        print("adding " + str(nodes[len(nodes) - 1].x) + ", " + str(nodes[len(nodes) - 1].y))
        # print("adding " + str(nodes[len(nodes) - 1].x) + ", " + str(nodes[len(nodes) - 1].y) + " g: " + str(nodes[len(nodes) - 1].g) + " h: " + str(nodes[len(nodes) - 1].h) + " f: " + str(nodes[len(nodes) - 1].f))
        nodes.sort(key=lambda x: x.f, reverse=True)

    if (parentNode.x, parentNode.y - 1) in coordinates:
        print("Changing G value")
        for ex in nodes:
            if ex.x == parentNode.x and ex.y == parentNode.y - 1:
                if ex.g > parentNode.g + 1:
                    ex.g = parentNode.g + 1
                    ex.f = ex.g + calculateH(parentNode.x, parentNode.y - 1)
                    ex.parent = parentNode.number
                    nodes.sort(key=lambda x: x.f, reverse=True)
                break

    elif (parentNode.x, parentNode.y - 1) not in explored and (parentNode.x,
                                                               parentNode.y - 1) not in obstacles and parentNode.x > 0 and parentNode.x <= gridX and parentNode.y - 1 > 0 and parentNode.y - 1 <= gridY:
        nodes.append(
            node(counter, parentNode.x, parentNode.y - 1, parentNode.g + 1,
                 calculateH(parentNode.x, parentNode.y - 1),
                 parentNode.number))
        coordinates.append((parentNode.x, parentNode.y - 1))
        counter += 1
        print("adding " + str(nodes[len(nodes) - 1].x) + ", " + str(nodes[len(nodes) - 1].y))
        # print("adding " + str(nodes[len(nodes) - 1].x) + ", " + str(nodes[len(nodes) - 1].y) + " g: " + str(nodes[len(nodes) - 1].g) + " h: " + str(nodes[len(nodes) - 1].h) + " f: " + str(nodes[len(nodes) - 1].f))
        nodes.sort(key=lambda x: x.f, reverse=True)

    if (parentNode.x + 1, parentNode.y - 1) in coordinates:
        print("Changing G value")
        for ex in nodes:
            if ex.x == parentNode.x + 1 and ex.y == parentNode.y - 1:
                if ex.g > parentNode.g + 1:
                    ex.g = parentNode.g + 1
                    ex.f = ex.g + calculateH(parentNode.x + 1, parentNode.y - 1)
                    ex.parent = parentNode.number
                    nodes.sort(key=lambda x: x.f, reverse=True)
                break

    elif (parentNode.x + 1, parentNode.y - 1) not in explored and (parentNode.x + 1,
                                                                   parentNode.y - 1) not in obstacles and parentNode.x + 1 > 0 and parentNode.x + 1 <= gridX and parentNode.y - 1 > 0 and parentNode.y - 1 <= gridY:
        nodes.append(node(counter, parentNode.x + 1, parentNode.y - 1, parentNode.g + 1.4,
                          calculateH(parentNode.x + 1, parentNode.y - 1), parentNode.number))
        coordinates.append((parentNode.x + 1, parentNode.y - 1))
        counter += 1
        print("adding " + str(nodes[len(nodes) - 1].x) + ", " + str(nodes[len(nodes) - 1].y))
        # print("adding " + str(nodes[len(nodes) - 1].x) + ", " + str(nodes[len(nodes) - 1].y) + " g: " + str(nodes[len(nodes) - 1].g) + " h: " + str(nodes[len(nodes) - 1].h) + " f: " + str(nodes[len(nodes) - 1].f))
        nodes.sort(key=lambda x: x.f, reverse=True)

class Grid:

    def __init__(self, root, width, height, numWidth, numHeight, line_width):
        self.totalDistance = 0
        self.grid = []

        self.visited = []
        for y in range(numHeight):
            tmp = []
            tmpV = []
            for i in range(numWidth):
                tmp.append(' ')
                tmpV.append(0)
            self.grid.append(tmp)
            self.visited.append(tmpV)

        self.root = root
        self.width = width
        self.height = height
        self.numWidth = numWidth
        self.numHeight = numHeight
        self.line_width = line_width

        self.c = Canvas(root, width=width + 1, height=height + 1, bg="black", highlightthickness=0)
        for i in range(numWidth + 1):
            xy = [(i * width / numWidth, 0), (i * width / numWidth, height)]
            self.c.create_line(xy, width=line_width, fill='white')
        for i in range(numHeight + 1):
            xy = [(0, i * height / numHeight), (width, i * height / numHeight)]
            self.c.create_line(xy, width=line_width, fill='white')

    def setStart(self, x, y):
        self.grid[y][x] = 's'
        self.startX = x
        self.startY = y
        self.startPos = [x, y]
        self.visited[y][x] = 1

        self.prvX, self.prvY = x, y

    def setWall(self, x, y):
        self.grid[y][x] = 'w'
        self.wall = self.c.create_rectangle((x * self.width / self.numWidth) + self.line_width,
                                            (y * self.height / self.numHeight) + self.line_width,
                                            ((x + 1) * self.width / self.numWidth),
                                            ((y + 1) * self.height / self.numHeight),
                                            fill='#787878', outline="") #gray

    def setLocation(self, x, y):
        self.location = self.c.create_rectangle((x * self.width / self.numWidth) + self.line_width,
                                                (y * self.height / self.numHeight) + self.line_width,
                                                ((x + 1) * self.width / self.numWidth),
                                                ((y + 1) * self.height / self.numHeight),
                                                fill='#FF3A3A', outline="") #red

        xy = [((self.prvX * self.width / self.numWidth) + (self.width / self.numWidth) / 2,
               (self.prvY * self.height / self.numHeight) + (self.height / self.numHeight) / 2), (
                  (x * self.width / self.numWidth) + (self.width / self.numWidth) / 2,
                  (y * self.height / self.numHeight) + (self.height / self.numHeight) / 2)]
        self.line = self.c.create_line(xy, width=2, fill='black') #line
        self.totalDistance += math.sqrt(math.pow(y - self.prvY, 2) + math.pow(x - self.prvX, 2))
        self.prvX = x
        self.prvY = y

    def fastestPath(self):
        while goal not in nodes:
            current = nodes[len(nodes) - 1]
            # print("checking " + str(current.x) + ", "+ str(current.y) + " F: " + str(current.f))
            print("checking: " + str(current.x) + ", " + str(current.y) + " number: " + str(
                current.number) + " parent: " + str(current.parent))
            exploredNodes.append(current)
            explored.append((current.x, current.y))

            nodes.remove(nodes[len(nodes) - 1])
            coordinates.remove((current.x, current.y))

            if (current.x, current.y) == (goal.x, goal.y):
                break
            else:
                checkNeighbors(current)

        for x in path:
            print(str(x.x) + ", " + str(x.y))

        for node in exploredNodes:
            if node.x == goalX and node.y == goalY:
                print("added goal to path")
                path.append(node)
                break

        for x in path:
            print(str(x.x) + ", " + str(x.y))

        print("creating path")

        while path[len(path) - 1].parent != 0:
            current = path[len(path) - 1]
            for node in exploredNodes:
                # print("number: " + str(node.number))
                if node.number == current.parent:
                    path.append(node)
                    # print("Adding: " + str(node.x) + ", " + str(node.y))
                    print("adding to path " + str(node.x) + ", " + str(node.y) + " number: " + str(
                        node.number) + " parent: " + str(node.parent))
                    current = node
                    break
        path.append(start)

        print("Start: " + str(startX) + ", " + str(startY))
        print("Goal: " + str(goalX) + ", " + str(goalY))

        for x in path:
            print(str(x.x) + ", " + str(x.y))
            self.setLocation(x.x - 1, 100 - x.y)
            # time.sleep(0.1)
            win.update()

        for i in path:
            for j in exploredNodes:
                if i == j:
                    exploredNodes.remove(j)

        for i in nodes:
            self.c.create_rectangle(((i.x - 1) * self.width / self.numWidth) + self.line_width,
                                    ((100 - i.y) * self.height / self.numHeight) + self.line_width,
                                    (i.x * self.width / self.numWidth),
                                    ((101 - i.y) * self.height / self.numHeight),
                                    fill='#FFCF3A', outline="")
        for i in exploredNodes:
            self.c.create_rectangle(((i.x - 1) * self.width / self.numWidth) + self.line_width,
                                    ((100 - i.y) * self.height / self.numHeight) + self.line_width,
                                    (i.x * self.width / self.numWidth),
                                    ((101 - i.y) * self.height / self.numHeight),
                                    fill='#FF843A', outline="")

        self.c.create_rectangle(((startX - 1) * self.width / self.numWidth) + self.line_width,
                                ((100 - startY) * self.height / self.numHeight) + self.line_width,
                                (startX * self.width / self.numWidth),
                                ((101 - startY) * self.height / self.numHeight),
                                fill='blue', outline="")
        self.c.create_rectangle(((goalX - 1) * self.width / self.numWidth) + self.line_width,
                                ((100 - goalY) * self.height / self.numHeight) + self.line_width,
                                (goalX * self.width / self.numWidth),
                                ((101 - goalY) * self.height / self.numHeight),
                                fill='green', outline="")
        print("DONE", "total distance =", self.totalDistance)

    def printGrid(self):
        for i in range(len(self.grid)):
            for j in range(len(self.grid[0])):
                if (self.grid[i][j] == ' '):
                    print('-', end="")
                else:
                    print(self.grid[i][j], end="")
            print()

    def getWidget(self):
        return self.c


#A STAR SETUP
coordinates = []
explored = []
exploredNodes = []
nodes = []
path = []
obstacles = [(2, 3), (2, 4), (2, 5), (2, 6), (2, 7), (2, 8), 
            (3, 6), (4, 6), (5, 6), (6, 6), (7, 6), (7, 7), (7, 8),
            (7, 9), (4, 2), (4, 3), (6, 2), (7, 2), (8, 2), (9, 2), (9, 3), (9, 4), (9, 5)]

for i in range(3333):
    while True:
        check = True
        block = (int(randrange(1, 100)), int(randrange(1, 100)))
        for coord in obstacles:
            if block == coord:
                check = False
                break
        if check:
            obstacles.append(block)
            break

counter = 0
gridX = 100
gridY = 100
startX = randrange(gridX) + 1
startY = randrange(gridY) + 1
goalX = randrange(gridX) + 1
goalY = randrange(gridY) + 1
# make sure start isnt in wall
while True:
    check = True
    for i in obstacles:
        if startX == i[0] and startY == i[1]:
            check = False
            break
    if check:
        break
    startX = randrange(gridX) + 1
    startY = randrange(gridY) + 1
# make sure goal isnt in wall or start
while True:
    check = True
    for i in obstacles:
        if goalX == startX and goalY == startY:
            check = False
            break
        if goalX == i[0] and goalY == i[1]:
            check = False
            break
    if check:
        break
    goalX = randrange(gridX) + 1
    goalY = randrange(gridY) + 1

start = node(counter, startX, startY, 0, calculateH(startX, startY), counter)
nodes.append(start)
coordinates.append((startX, startY))
counter += 1

goal = node(counter, goalX, goalY, 0, calculateH(startX, startY), counter)
print(node)
counter += 1


#GUI SETUP
width = 800
height = 800
line_width = 1
dim = str(width + line_width) + "x" + str(height + line_width)

win = tk.Tk()
win.title("A star")
win.geometry(dim)
win.configure(bg='black')
g = Grid(win, width, height, 100, 100, line_width)

g.setStart(startX - 1, 100 - startY)
for i in obstacles:
    g.setWall(i[0] - 1, 100 - i[1])

g.getWidget().pack()

#PLOT A STAR
g.fastestPath()
for i in obstacles:
    g.setWall(i[0] - 1, 100 - i[1])
print('START: [' + str(startX) + ', ' + str(startY) + ']')
print('GOAL: [' + str(goalX) + ', ' + str(goalY) + ']')
win.mainloop()
