from asyncio.windows_events import NULL
import json
import numpy as np
import cv2
import colorsys
import random
import os
import shutil
from sklearn.neighbors import RadiusNeighborsClassifier

"""This runs with traincolors.json and testcolors.json which depends on the train and test folders
    hexcolors.json are the colors I webscrapped converted to rgb """

def train_color_data():
    """hsv values are H: 0-179, S: 0-255, V:0-255
    dictionary of bounds : [color]: [array of np.array]
    generating train data by collect points within the multiple ranges"""

    dic_lowerbounds = {"red": [np.array([0, 138, 60],np.uint8), np.array([171, 138, 60],np.uint8)], "orange": [np.array([10, 150, 165],np.uint8)], \
                    "yellow": [np.array([24, 138, 130],np.uint8)], "green": [np.array([37, 138, 60],np.uint8)], "blue": [np.array([90, 138, 60],np.uint8)],\
                    "purple": [np.array([130, 138, 60],np.uint8)], "gray": [np.array([0, 0, 60],np.uint8)], \
                    "black": [np.array([0, 0, 0],np.uint8)], "white": [np.array([0, 0, 214],np.uint8)], "brown": [np.array([9,78,60],np.uint8)]
                    } 

    dic_upperbounds = {"red": [np.array([9, 255, 255],np.uint8), np.array([179, 255, 255],np.uint8)], "orange": [np.array([23, 255, 255],np.uint8)], \
                    "yellow": [np.array([30, 255, 255],np.uint8)], "green": [np.array([77, 255, 255],np.uint8)], "blue": [np.array([115, 255, 255],np.uint8)],\
                    "purple": [np.array([165, 255, 255],np.uint8)], "gray": [np.array([179, 50, 130],np.uint8)], \
                    "black": [np.array([179, 255, 45],np.uint8)], "white": [np.array([179, 26, 255],np.uint8)], "brown": [np.array([23,255,160])]}

    img_h, img_w, img_c = (100,100,3)
    dirct = 'train'
    wrtcdataf = dirct + "colors.json"
    mydata ={}

    '''recreating hsv database'''

    for color in dic_lowerbounds:
        #enter train folder
        if os.path.exists(dirct) is False:
            os.makedirs(dirct)
        os.chdir(dirct)

        #delete old color folder make new folder
        if os.path.exists(color) is True:
            shutil.rmtree(color) 

        os.makedirs(color)
        os.chdir(color)

        mydata[color] = []
        for datagen in range(0,200):
            #for multiple thresholds

            for sett in range(0,len(dic_lowerbounds[color])):
                H = float(random.randint(dic_lowerbounds[color][sett][0],dic_upperbounds[color][sett][0])/179  )
                S = float(random.randint(dic_lowerbounds[color][sett][1],dic_upperbounds[color][sett][1])/255  )
                V = float(random.randint(dic_lowerbounds[color][sett][2],dic_upperbounds[color][sett][2])/255  )

                r,g,b = colorsys.hsv_to_rgb(H,S,V)
                R,G,B = int(r*255),int(g*255),int(b*255)
                mydata[color].append([R,G,B])

                newcolorshade = np.full((img_h,img_w,img_c), [B,G,R],dtype = np.uint8)
                cv2.imwrite(str([R,G,B]) + ".jpg" ,newcolorshade)

        os.chdir(os.path.abspath(os.pardir))

    json_object = json.dumps(mydata)
    with open(wrtcdataf, "w") as outfile:
        outfile.write(json_object)


"""will update the datafile based on the uploaded rgb images in the color folder"""
def update_colordatafile(dirct):
    tst_data = {}
    colorsdirect = ["red", "orange", "yellow", "green", "blue", "purple", "brown", "gray", "white", "black"]
    wrtcdataf = dirct + "colors.json"

    for clrlabel in colorsdirect:
        #enter test or train directory
        if os.path.exists(dirct) is True:
            os.chdir(dirct)
            if os.path.exists(clrlabel) is True:
                tst_data[clrlabel] = []
                l=os.listdir(clrlabel)
                filesnoext=[x.split('.')[0] for x in l]

                for filenam in filesnoext:
                    colorval = json.loads(filenam)
                    tst_data[clrlabel].append(colorval)
            
            os.chdir(os.path.abspath(os.pardir))

    json_object = json.dumps(tst_data)
    with open(wrtcdataf, "w") as outfile:
        outfile.write(json_object)


"""classifying data and running an accuracy check based on test data """
def color_rknn_classify(traindatafile,testdatafile):

  shadesofcolordata = json.load(open(traindatafile))
  testdata = json.load(open(testdatafile))

  xtest = []
  ylabeltest = []
  xdata = []
  ylabel = []

  kradius = 100
  for colorlabel in shadesofcolordata:
    for shadevalue in shadesofcolordata[colorlabel]:
      xdata.append(shadevalue) 
      ylabel.append(colorlabel)

  colorneigh = RadiusNeighborsClassifier(radius = kradius, weights = "distance",outlier_label = "gray")
  colorneigh.fit(xdata, ylabel)

  for colorlabel in testdata:
     xtest = []
     ylabeltest = []
     for shadevalue in testdata[colorlabel]:
      xtest.append(shadevalue) 
      ylabeltest.append(colorlabel)

  
     ypredict = colorneigh.predict(xtest)
     print(f"{colorlabel}:  {colorneigh.score(xtest, ylabeltest)}")



if __name__ == "__main__":

    newTraindata = False
    newdatadirct = NULL
    knnTest = True

    if newTraindata is True:
        train_color_data()
    if newdatadirct is not NULL:
        print(newdatadirct)
        update_colordatafile(newdatadirct)
    if knnTest is True:
        color_rknn_classify("traincolors.json","testcolors.json")





