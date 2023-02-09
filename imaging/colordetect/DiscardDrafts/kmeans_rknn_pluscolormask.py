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
    # Transform image into 'data':
    # Define the algorithm termination criteria (the maximum number of iterations and/or the desired accuracy):
    # In this case the maximum number of iterations is set to 20 and epsilon = 1.0 , subject to change
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)

    #making a numpy with 4 dimensions
    fivedimensionallist = []
    for xlocation in range(0,row):
            for ylocation in range(0,col):
                b,g,r = image[xlocation][ylocation]
                radius = round((((row//2 - xlocation))**2 + ((col//2 - ylocation))**2)**0.5)
                fivedimensionallist.append([b,g,r,radius])
    reshapeimgarray = np.array(fivedimensionallist, dtype=np.uint8)
    reshapeimgarray = np.float32(reshapeimgarray)
    onecolmnarry = reshapeimgarray.shape
 #   print(fivedimensionallist)

    # Apply K-means clustering algorithm:
    retwlocation, labelwlocation, centerwlocation = cv2.kmeans(reshapeimgarray, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    centerwlocation = np.uint8(centerwlocation)
    centroid_mask = np.copy(centerwlocation)
    centroid_mask = np.delete(centroid_mask,-1,1)
   # print(centroid_mask)

    result = centerwlocation[labelwlocation.flatten()]
    result = result.reshape(onecolmnarry)
    result = np.delete(result,-1,1)
#    result = np.delete(result,-1,1)   #if we used 5 dimensions include this to return the array back to 3 channels

    result = result.reshape(image.shape)

    ind=np.argsort(centerwlocation[:,-1]) #sort by least distance from center to most
    centerwlocation = centerwlocation[ind]
    cv2.imshow('reconfig img', result)
    cv2.waitKey(0)


    return result, centerwlocation


def letter_shape_mask(reconfiguredimg, ordered_centroids):
  color_range = 1
  letter_upper_rgb = np.array(ordered_centroids[0][0:3])
  letter_low_rgb = letter_upper_rgb.copy()
  bckgrd_upper_rgb = np.array(ordered_centroids[2][0:3])
  bckgrd_low_rgb = bckgrd_upper_rgb.copy()

  letter_upper_rgb[letter_upper_rgb <255] += color_range
  letter_low_rgb[letter_low_rgb >0] -= color_range
  bckgrd_upper_rgb[bckgrd_upper_rgb <255] += color_range
  bckgrd_low_rgb[bckgrd_low_rgb >0] -= color_range  
#  print(letter_low_rgb,letter_upper_rgb)    

  #letter mask
  lettercolormask = cv2.inRange(reconfiguredimg, letter_low_rgb,letter_upper_rgb)
  letterresult = cv2.bitwise_and(reconfiguredimg,reconfiguredimg,mask=lettercolormask)
  letterresult = cv2.bitwise_not(letterresult)
  letterresult = cv2.cvtColor(letterresult, cv2.COLOR_BGR2GRAY)
  ret,letterfiltered = cv2.threshold(letterresult,254,255,cv2.THRESH_BINARY)

  #shape mask by inverting background noise
  shapecolormask = cv2.inRange(threemaskimage, bckgrd_low_rgb,bckgrd_upper_rgb)
  shaperesult = cv2.bitwise_and(reconfiguredimg,reconfiguredimg,mask=shapecolormask)
  shaperesult = cv2.bitwise_not(shaperesult)
  shaperesult = cv2.cvtColor(shaperesult, cv2.COLOR_BGR2GRAY)
  ret1,shapebw = cv2.threshold(shaperesult,254,255,cv2.THRESH_BINARY)
  shapefiltered = cv2.bitwise_not(shapebw)

  cv2.imshow('letter', letterfiltered)
  cv2.waitKey(0)
  cv2.imshow('shape', shapefiltered)
  cv2.waitKey(0)
  return letterfiltered, shapefiltered
  #  print(threemaskimage, '\n', centroid)

def color_rknn_classify(colordatafile,centroidlst, lettermsk, shapemsk):
  #needs to be done in hsv
  #consider removing saturation and only take into account h and v
  shadesofcolordata = json.load(open(colordatafile))
  centroid_value = np.copy(centroidlst)
  centroid_value = np.delete(centroid_value,-1,1)
  xdata = []
  ylabel = []
  kradius = 100
  for colorlabel in shadesofcolordata:
    for shadevalue in shadesofcolordata[colorlabel]:
#      xdata.append([shadevalue['R'],shadevalue['G'],shadevalue['B']])
      xdata.append([shadevalue['H'],shadevalue['S'],shadevalue['V']])
#      xdata.append([shadevalue['H'],shadevalue['V']])
      ylabel.append(colorlabel)

 # print(xdata)
  hv_centroid = []
  for rgbvalue in centroid_value:
    h,s,v = colorsys.rgb_to_hsv(rgbvalue[0]/255, rgbvalue[1]/255, rgbvalue[2]/255)
    print(h,s,v)
    color_hsv = [round(h*179), round(s*255), round(v*255)]
    print(color_hsv)
   # color_hsv = [round(h*179), round(v*255)]
    print(color_hsv)
    hv_centroid.append(color_hsv)


 # print(centroid_value[0].tolist())
  print(hv_centroid)
  colorneigh = RadiusNeighborsClassifier(kradius)
  colorneigh.fit(xdata, ylabel)
 # letter_predict = colorneigh.predict([centroid_value[0].tolist()])
 # shape_predict = colorneigh.predict([centroid_value[1].tolist()])
  letter_predict = colorneigh.predict([hv_centroid[0]])
  shape_predict = colorneigh.predict([hv_centroid[1]])
  print(letter_predict, shape_predict)
  colordatafile.close()

if __name__ == "__main__":
    
#--------------------------------------------------------------------------------------------------------------
#    datatestreport = {}     for scoring and report purposes ignore for now
#    notdetected = {"red":0, "orange":0, "yellow":0, "green":0, "blue":0, "purple":0, "gray":0, "white":0, "black":0, "brown":0}
#    test = 0
#--------------------------------------------------------------------------------------------------------------

    files = open("circleblurtestset1.json")  # json with the datatestset of the file location, and letter, shape
    #debugfiles = open("debugcircletestset1.json")
    datasample = json.load(files)

    #checking for accuracy an integer
    total = len(datasample)
    score = 0
    #have an overall for loop
    for eachfilelocation in datasample:
       # test +=1  IGNORE WAS FOR TESTING
       # datatestreport[test] = []
       # datatestreport[test].append(eachfilelocation) 
        image = cv2.imread(eachfilelocation)
        #image = cv2.imread("C:\\Users\\kirva\\Desktop\\forge\\winter2023\\blurtests\\circletestset1\\circleblursample-17whitered.jpg")
        cv2.imshow('og', image)
        cv2.waitKey(0)
        rgbimage = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
      # rgbimage = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
      #  cv2.imshow('res', rgbimage)
      #  cv2.waitKey(0)
        threemaskimage, centroid = color_quantization(rgbimage,3)
        print(centroid)
      #  cv2.imshow('res', threemaskimage)
      #  cv2.waitKey(0)
        lettermask, shapemask = letter_shape_mask(threemaskimage,centroid)
        #next is classification
        classifiedcolor = color_rknn_classify('sample.json',centroid,lettermask,shapemask)
        #classifiedcolor is a dictionary with key 'letter' and 'shape'
        break

#-------------------------------------------------------------------------------------
# scoring to be implemented for accuracy
  #  print("kmeans detects both colors", score)
  #  print(notdetected)
    

  #  json_object = json.dumps(datatestreport, indent=4, sort_keys=True)
  #  with open("circlecircletestset1report.json", "w") as outfile:
  #      outfile.write(json_object)



        


