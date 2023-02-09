import json
import numpy as np
import cv2
import colorsys
import random


mydata ={}

dic_lowerbounds = {"red": [np.array([0, 138, 163],np.uint8), np.array([171, 138, 163],np.uint8)], "orange": np.array([11, 138, 163],np.uint8), \
                   "yellow": np.array([22, 138, 163],np.uint8), "green": np.array([36, 138, 163],np.uint8), "blue": np.array([80, 138, 163],np.uint8),\
                   "purple": np.array([132, 138, 163],np.uint8), "gray": np.array([0, 0, 68],np.uint8), \
                   "black": np.array([0, 0, 0],np.uint8), "white": np.array([0, 0, 214],np.uint8), "brown":np.array([9,104,70],np.uint8)
                   } 

dic_upperbounds = {"red": [np.array([10, 255, 255],np.uint8), np.array([179, 255, 255],np.uint8)], "orange": np.array([21, 255, 255],np.uint8), \
                   "yellow": np.array([35, 255, 255],np.uint8), "green": np.array([70, 255, 255],np.uint8), "blue": np.array([128, 255, 255],np.uint8),\
                   "purple": np.array([170, 255, 255],np.uint8), "gray": np.array([179, 50, 140],np.uint8), \
                   "black": np.array([179, 255, 39],np.uint8), "white": np.array([179, 26, 255],np.uint8), "brown":np.array([25,255,128])}


for color in dic_lowerbounds:
    mydata[color] = []
    for datagen in range(0,500):
        if color == "red":
            for sett in range(0,2):

                H = random.randint(dic_lowerbounds[color][sett][0],dic_upperbounds[color][sett][0]) *2 
                S = int(random.randint(dic_lowerbounds[color][sett][1],dic_upperbounds[color][sett][1])/255 *100)
                V = int(random.randint(dic_lowerbounds[color][sett][2],dic_upperbounds[color][sett][2])/255 *100)
              #  print(color, H,S,V)
                mydata[color].append([int(H/2),int(S/100*255),int(V/100*255)])
                
        else:     
            H = random.randint(dic_lowerbounds[color][0],dic_upperbounds[color][0]) *2 
            S = int(random.randint(dic_lowerbounds[color][1],dic_upperbounds[color][1])/255 *100)
            V = int(random.randint(dic_lowerbounds[color][2],dic_upperbounds[color][2])/255 *100)
        #    print(color, H,S,V)
            mydata[color].append([int(H/2),int(S/100*255),int(V/100*255)])







#for eachkey in mydata:
 #   print(mydata[eachkey])
json_object = json.dumps(mydata, indent=4)
with open("sample.json", "w") as outfile:
    outfile.write(json_object)
