from asyncio.windows_events import NULL
import json
import numpy as np
import cv2
import colorsys
import random
import os
import shutil
import splitfolders
from sklearn.neighbors import RadiusNeighborsClassifier

"""This runs with traincolors.json and testcolors.json which depends on the train and test folders
    hexcolors.json are the colors I webscrapped converted to rgb """

def train_color_data():
    """hsv values are H: 0-179, S: 0-255, V:0-255
    dictionary of bounds : [color]: [array of np.array]
    generating train data by collect points within the multiple ranges"""

    dic_lowerbounds = {"red": [np.array([0, 200, 80],np.uint8), np.array([174, 200, 80],np.uint8)], "orange": [np.array([10, 190, 160],np.uint8)], \
                    "yellow": [np.array([24, 190, 130],np.uint8)], "green": [np.array([43, 190, 80],np.uint8)], "blue": [np.array([90, 190, 80],np.uint8)],\
                    "purple": [np.array([130, 190, 80],np.uint8)], "gray": [np.array([0, 0, 77],np.uint8)], \
                    "black": [np.array([0, 0, 0],np.uint8)], "white": [np.array([0, 0, 214],np.uint8)], "brown": [np.array([13,130,58],np.uint8)]
                    } 

    dic_upperbounds = {"red": [np.array([5, 255, 255],np.uint8), np.array([179, 255, 255],np.uint8)], "orange": [np.array([18, 255, 255],np.uint8)], \
                    "yellow": [np.array([28, 255, 255],np.uint8)], "green": [np.array([74, 255, 255],np.uint8)], "blue": [np.array([115, 255, 255],np.uint8)],\
                    "purple": [np.array([165, 255, 255],np.uint8)], "gray": [np.array([179, 8, 130],np.uint8)], \
                    "black": [np.array([179, 255, 20],np.uint8)], "white": [np.array([179, 26, 255],np.uint8)], "brown": [np.array([18,255,102])]}

    img_h, img_w, img_c = (100,100,3)
    dirct = 'train'
    wrtcdataf = dirct + "colors.json"
    mydata ={}

    '''recreating hsv database'''

    if os.path.exists(dirct) is False:
            os.makedirs(dirct)
    os.chdir(dirct)

    for color in dic_lowerbounds:
        #enter train folder
        color = "red"
        #delete old color folder make new folder
        if os.path.exists(color) is True:
            shutil.rmtree(color) 

        os.makedirs(color)
        os.chdir(color)

        mydata[color] = []
        for datagen in range(0,1000):
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
        print("get cwd", os.getcwd())
        os.chdir("../")
        break
        

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
            
            #os.chdir(os.path.abspath(os.pardir))
            print("get cwd", os.getcwd())
            os.chdir("../")

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


def Gen_Noisy_Img(ImgDirectory):
 #   ImgDirectory = "test" # example
    colorsdirect = ["red", "orange", "yellow", "green", "blue", "purple", "brown", "gray", "white", "black"]
 
    if os.path.exists(ImgDirectory) is True:
        os.chdir(ImgDirectory)
        print("get cwd", os.getcwd())
        for clrlabel in colorsdirect:
            if os.path.exists(clrlabel) is True:
                colorlst=os.listdir(clrlabel)
                os.chdir(clrlabel)
                print("get cwd", os.getcwd())
                for clrfile in colorlst:
                    if clrfile.endswith("g.jpg"):
                        os.remove(os.path.join(os.getcwd(), clrfile))
                        continue
                    print(clrfile)
                    readfile = cv2.imread(clrfile)
                  #  readfile = cv2.cvtColor(readfile, cv2.COLOR_BGR2RGB)
                    # blur + brighten 

                    # define the contrast and brightness value
                    ksize =  (random.randint(50,80),random.randint(50,80))
                    blur = cv2.blur(readfile,ksize)

                    # call addWeighted function. use beta = 0 to effectively only operate on one image
                    cv2.imwrite(str(blur[0][0])+'.jpg',blur)
                os.chdir("../")
                






if __name__ == "__main__":

    newTraindata = False
    newdatadirct = NULL
    knnTest = False
    genMoreImg = NULL
    splitFolders = True

    if newTraindata is True:
        train_color_data()
    if genMoreImg is not NULL:
        print(genMoreImg)
        Gen_Noisy_Img(genMoreImg)
    if newdatadirct is not NULL:
        print(newdatadirct)
        update_colordatafile(newdatadirct)
    if knnTest is True:
        color_rknn_classify("traincolors.json","testcolors.json")
    if splitFolders is True:
        input_folder = os.path.join(os.getcwd(), "input") #path of your input folder
        output = os.path.join(os.getcwd(), "Database") ##where you want the split datasets saved. one will be created if it does not exist or none is set

        splitfolders.ratio(input_folder, output=output, seed=42, ratio=(.8, .1, .1))