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

def color_quantization(image, k):
    """Performs color quantization using K-means clustering algorithm"""
#    cv2.imshow('img', image)
#    cv2.waitKey(0)
    row,col,channel = image.shape
    # Transform image into 'data':
    data = np.float32(image).reshape((-1, 3))
#    print(data.shape)
#    print(type(data))
    # Define the algorithm termination criteria (the maximum number of iterations and/or the desired accuracy):
    # In this case the maximum number of iterations is set to 20 and epsilon = 1.0
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)

    #making a numpy with 5 dimensions
    fivedimensionallist = []

    for xlocation in range(0,row):
            for ylocation in range(0,col):
                b,g,r = image[xlocation][ylocation]
                fivedimensionallist.append([b,g,r,abs((row//2 - xlocation)),abs((col//2 - ylocation))])
    reshapeimgarray = np.array(fivedimensionallist, dtype=np.uint8)
    reshapeimgarray = np.float32(reshapeimgarray)
    singlefileline = reshapeimgarray.shape


    # Apply K-means clustering algorithm:
    ret, label, center = cv2.kmeans(data, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    retwlocation, labelwlocation, centerwlocation = cv2.kmeans(reshapeimgarray, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)


   # centerwlocation = np.uint8(centerwlocation)

 #   print(centerwlocation)
    print(labelwlocation.shape)


    centroid_mask =[]
    for eachcentroid in centerwlocation:
            centroid_mask.append(np.array([eachcentroid[0],eachcentroid[1],eachcentroid[2]],dtype=np.uint8))
    centroid_mask = np.array(centroid_mask)
  #  print('yo', centroid_mask)
    # At this point we can make the image with k colors
    # Convert center to uint8:
    centerwlocation = np.uint8(centerwlocation)
    center = np.uint8(center)
    #centerwlocation = np.uint8(centerwlocation).tolist()
    #print(centerwlocation)
    #print(centerwlocation)

    #orderedcenterwlocation = [centerwlocation[i][0:3] for i in range(0,3)]
   
    #print("ahhh", orderedcenterwlocation)
#    centerwlocation = np.array(centerwlocation)
#    print(ret1, '\n', label1, '\n', center1, '\n')
    # Replace pixel values with their center value:
    result1 = center[label.flatten()]
    print(result1.shape)
    result1 = result1.reshape(image.shape)
    result = centerwlocation[labelwlocation.flatten()]
    print('help', result1.shape)
    result = result.reshape(singlefileline)
    result = np.delete(result,-1,1)
 # print(result.shape)
    result = np.delete(result,-1,1)
    print('help', result.shape)

    print(result)
    result = result.reshape(image.shape)

  #  result1 = center1[label1.flatten()]
  #  print("what's happening \n", type(label))
  #  centerwlocation = centerwlocation.sort(key = lambda centerwlocation:centerwlocation[-1])
    ind=np.argsort(centerwlocation[:,-1])
    centerwlocation = centerwlocation[ind]
    print(centerwlocation)
   # cv2.imshow('res', result1)
   # cv2.waitKey(0)
    cv2.imshow('res', result)
    cv2.waitKey(0)


    return result1, centerwlocation


def findrgbpixelcolor(arg1,dataarg2, colors, minimum,colorname):
    b,g,r = arg1
    B,G,R = dataarg2
  #  print(round(H*179),round(S*255),round(255*V))
 #   refH, refS, refV = dataarg2
    d =  ((abs(b-B))**2 + (abs(g-G))**2 + (abs(r-R))**2)**0.5
  #  d =  ((abs(b-B)) + (abs(g-G)) + (abs(r-R)))
    #print( b, g, r, B, G, R)
   # print("argument", arg1)
    if (d<= minimum):
        minimum = d
        colorname = colors
    return (colorname,minimum)

if __name__ == "__main__":
    image = cv2.imread("C:\\Users\\kirva\\Desktop\\forge\\winter2023\\blurtests\\circletestset1\\circleblursample-17whitered.jpg")
    cv2.imshow('res', image)
    cv2.waitKey(0)
    rgbimage = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
   # rgbimage = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    cv2.imshow('res', rgbimage)
    cv2.waitKey(0)
    threemaskimage, centroid = color_quantization(rgbimage,3)
    print(centroid)
    cv2.imshow('res', threemaskimage)
    cv2.waitKey(0)


    # colordetection = color_mask(threemaskimage, centroid)
    # colordetection = "letter": ["color", "npimg"]

    print(threemaskimage)
    letter_upper_rgb = np.array(centroid[2][0:3])
   # letter_upper_rgb[0] = 255
    letter_low_rgb = letter_upper_rgb.copy()
  #  shape_upper_rgb = np.array(centroid[1][0:3])
  #  shape_low_rgb = shape_upper_rgb.copy()

  #  letter_upper_rgb[letter_upper_rgb <=255] += 1
    letter_low_rgb[letter_low_rgb >0] -= 1
  #  shape_upper_rgb[letter_upper_rgb <255] += 1
  #  shape_low_rgb[letter_low_rgb >0] -= 1   
    print(letter_low_rgb,letter_upper_rgb)     
    lettercolormask = cv2.inRange(threemaskimage, letter_low_rgb,letter_upper_rgb)
    letterresult = cv2.bitwise_and(threemaskimage,threemaskimage,mask=lettercolormask)
    letterresult = cv2.bitwise_not(letterresult)
    letterresult = cv2.cvtColor(letterresult, cv2.COLOR_BGR2GRAY)
    ret,letterbw = cv2.threshold(letterresult,254,255,cv2.THRESH_BINARY)
    letterbw = cv2.bitwise_not(letterbw)
    cv2.imshow('res3000', letterbw)


    

    cv2.waitKey(0)
  #  print(threemaskimage, '\n', centroid)

    


