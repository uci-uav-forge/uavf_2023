from asyncio.windows_events import NULL
from turtle import color
import numpy as np
import cv2
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from collections import Counter
import colorsys
from operator import itemgetter
import json
from sklearn.neighbors import RadiusNeighborsClassifier

def color_quantization(image, k):
    """Performs color quantization using K-means clustering algorithm"""
    row,col,channel = image.shape
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)

    fivedimensionallist = []
    for xlocation in range(0,row):
        for ylocation in range(0,col):
            b,g,r = image[xlocation][ylocation]
            radius = round((((row//2 - xlocation))**2 + ((col//2 - ylocation))**2)**0.5)
            fivedimensionallist.append([b,g,r,1/(radius+1)*10000])

    reshapeimgarray = np.array(fivedimensionallist, dtype=np.uint8)

    reshapeimgarray = np.float32(reshapeimgarray)
    onecolmnarry = reshapeimgarray.shape

    # Apply K-means clustering algorithm:
    retwlocation, labelwlocation, centerwlocation = cv2.kmeans(reshapeimgarray, k, None, criteria, 50, cv2.KMEANS_RANDOM_CENTERS)

    
    print(centerwlocation)
    centerwlocation = np.uint8(centerwlocation)

    result = centerwlocation[labelwlocation.flatten()]
    result = result.reshape(onecolmnarry)
    result = np.delete(result,-1,1)  

    print(centerwlocation)

    result = result.reshape(image.shape)

    ind=np.argsort(centerwlocation[:,-1]) 
    centerwlocation = centerwlocation[ind]
    print(centerwlocation)
 
    


    return result, centerwlocation

"""color masking from kmeans------------------------------------------------------------------------------------------------------------"""
def letter_shape_mask(reconfiguredimg, ordered_centroids):
  color_range = 1
  letter_upper_rgb = np.array(ordered_centroids[0][0:3]) #used to be 0
  letter_low_rgb = letter_upper_rgb.copy()
  bckgrd_upper_rgb = np.array(ordered_centroids[2][0:3])  #used to be 2
  bckgrd_low_rgb = bckgrd_upper_rgb.copy()

  letter_upper_rgb[letter_upper_rgb <255] += color_range
  letter_low_rgb[letter_low_rgb >0] -= color_range
  bckgrd_upper_rgb[bckgrd_upper_rgb <255] += color_range
  bckgrd_low_rgb[bckgrd_low_rgb >0] -= color_range     


  lettercolormask = cv2.inRange(reconfiguredimg, letter_low_rgb,letter_upper_rgb)
  letterresult = cv2.bitwise_and(reconfiguredimg,reconfiguredimg,mask=lettercolormask)
  letterresult = cv2.bitwise_not(letterresult)
  letterresult = cv2.cvtColor(letterresult, cv2.COLOR_BGR2GRAY)
  ret,letterfiltered = cv2.threshold(letterresult,254,255,cv2.THRESH_BINARY)

  shapecolormask = cv2.inRange(threemaskimage, bckgrd_low_rgb,bckgrd_upper_rgb)
  shaperesult = cv2.bitwise_and(reconfiguredimg,reconfiguredimg,mask=shapecolormask)
  shaperesult = cv2.bitwise_not(shaperesult)
  shaperesult = cv2.cvtColor(shaperesult, cv2.COLOR_BGR2GRAY)
  ret1,bckbw = cv2.threshold(shaperesult,254,255,cv2.THRESH_BINARY)



  shapefiltered = cv2.bitwise_not(bckbw)


  return letterfiltered, shapefiltered


"""comparing rgb values to colorbase set ---------------------------------------------------------------------------------------------------"""
def color_rknn_classify(colordatafile,centroidlst, lettermsk, shapemsk):

  shadesofcolordata = json.load(open(colordatafile))
  centroid_value = np.copy(centroidlst)
  centroid_value = np.delete(centroid_value,-1,1)
  xdata = []
  ylabel = []
  kradius = 80
  for colorlabel in shadesofcolordata:
    for shadevalue in shadesofcolordata[colorlabel]:

      xdata.append(shadevalue) 
      ylabel.append(colorlabel)


  colorneigh = RadiusNeighborsClassifier(radius = kradius, weights ="distance")
  colorneigh.fit(xdata, ylabel)
  letter_predict = colorneigh.predict([centroid_value[-1]]) 
  shape_predict = colorneigh.predict([centroid_value[1]])   
  print('letter:', letter_predict,'shape:', shape_predict)

  return {"letter": letter_predict,"shape":shape_predict}



if __name__ == "__main__":
    
    """ reads the json of test photos, kmeans, mask, classify"""
    score = 0
    datatestreport = {}
    test = 0
    files = open("circleblurtestset1.json") 
   
    datasample = json.load(files)

    for eachfilelocation in datasample:
 
        datatestreport[test] = []
        datatestreport[test].append(eachfilelocation)
        datatestreport[test].append(datasample[eachfilelocation])
        print('new photo: ', datasample[eachfilelocation])
        rightshape , rightletter = datasample[eachfilelocation]['shape'],datasample[eachfilelocation]['letter']
        image = cv2.imread(eachfilelocation)

        rgbimage = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


        threemaskimage, centroid = color_quantization(rgbimage,3)
        datatestreport[test].append(centroid.tolist())
        #image is reconstructed into three colors
        lettermask, shapemask = letter_shape_mask(threemaskimage,centroid)


        #next is classification
        classifiedcolor = color_rknn_classify('colordataset.json',centroid,lettermask,shapemask)

        #classifiedcolor is a dictionary with key 'letter' and 'shape'
        if (rightletter == classifiedcolor["letter"] and rightshape == classifiedcolor["shape"]):
          score += 1
          datatestreport[test].append(score)
          print('correct')
        else:
          datatestreport[test].append([classifiedcolor['letter'].tolist(),classifiedcolor['shape'].tolist()])
        test += 1
        
        
        
        
    """writes up accuracy report"""     
    print(score)
    json_object = json.dumps(datatestreport, indent=2)
    with open("rgbtest_singlerun.json", "w") as outfile:
        outfile.write(json_object)





        


