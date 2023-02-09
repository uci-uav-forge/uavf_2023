import json
import numpy as np
import cv2
import colorsys
import random


mydata ={}

dic_lowerbounds = {"red": [np.array([0, 100, 100],np.uint8), np.array([159, 100, 100],np.uint8)], "orange": np.array([10, 100, 150],np.uint8), \
                   "yellow": np.array([20, 100, 100],np.uint8), "green": np.array([36, 100, 100],np.uint8), "blue": np.array([80, 100, 100],np.uint8),\
                   "purple": np.array([132, 100, 100],np.uint8), "gray": np.array([0, 0, 50],np.uint8), \
                   "black": np.array([0, 0, 0],np.uint8), "white": np.array([0, 0, 231],np.uint8), "brown":np.array([10,115,51],np.uint8)
                   } 

dic_upperbounds = {"red": [np.array([9, 255, 255],np.uint8), np.array([179, 255, 255],np.uint8)], "orange": np.array([15, 255, 255],np.uint8), \
                   "yellow": np.array([35, 255, 255],np.uint8), "green": np.array([70, 255, 255],np.uint8), "blue": np.array([128, 255, 255],np.uint8),\
                   "purple": np.array([150, 255, 255],np.uint8), "gray": np.array([180, 18, 150],np.uint8), \
                   "black": np.array([179, 255, 40],np.uint8), "white": np.array([180, 18, 255],np.uint8), "brown":np.array([20,255,127])}



for color in dic_lowerbounds:
    mydata[color] = []
    for datagen in range(0,100):
        if color == "red":
            for sett in range(0,2):

                H = random.randint(dic_lowerbounds[color][sett][0],dic_upperbounds[color][sett][0]) *2 
                S = int(random.randint(dic_lowerbounds[color][sett][1],dic_upperbounds[color][sett][1])/255 *100)
                V = int(random.randint(dic_lowerbounds[color][sett][2],dic_upperbounds[color][sett][2])/255 *100)
                print(color, H,S,V)
                mydata[color].append([int(H/2),int(S/100*255),int(V/100*255)])
                
        else:     
            H = random.randint(dic_lowerbounds[color][0],dic_upperbounds[color][0]) *2 
            S = int(random.randint(dic_lowerbounds[color][1],dic_upperbounds[color][1])/255 *100)
            V = int(random.randint(dic_lowerbounds[color][2],dic_upperbounds[color][2])/255 *100)
            print(color, H,S,V)
            mydata[color].append([int(H/2),int(S/100*255),int(V/100*255)])

for eachkey in mydata:
    print(mydata[eachkey])
json_object = json.dumps(mydata, indent=4)
with open("sample.json", "w") as outfile:
    outfile.write(json_object)
