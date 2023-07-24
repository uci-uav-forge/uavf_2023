import numpy as np
import cv2
import json
from sklearn.neighbors import RadiusNeighborsClassifier


def color_quantization(image, k, num):
    """Performs color quantization using K-means clustering algorithm"""
    row,col,channel = image.shape
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)

    nparry_2dlist = []
    for xlocation in range(0,row):
        for ylocation in range(0,col):
            c1,c2,c3 = image[xlocation][ylocation]
            x, y = abs((row//2 - xlocation)), abs((col//2 - ylocation))
            radius = round((((row//2 - xlocation))**2 + ((col//2 - ylocation))**2)**0.5)
            
            nparry_2dlist.append([c1,c2, c3, radius])

    reshapeimgarray = np.array(nparry_2dlist, dtype=np.uint8)

    reshapeimgarray = np.float32(reshapeimgarray)
    onecolmnarry = reshapeimgarray.shape

    rets, labels, centroids = cv2.kmeans(reshapeimgarray, k, None, criteria, 50, cv2.KMEANS_RANDOM_CENTERS)            
            
    centroids = np.uint8(centroids)

    kmeans_result = centroids[labels.flatten()]
    kmeans_result = kmeans_result.reshape(onecolmnarry)
    kmeans_result = np.delete(kmeans_result,-1,1)  

    result = kmeans_result.reshape(image.shape)

    ind=np.argsort(centroids[:,-1]) 
    ordered_centroids = centroids[ind]

    resultrgb = result
  
 #   cv2.imwrite(str(num)+'kmeans.jpg', resultrgb)

    return result, ordered_centroids


"""comparing rgb values to colorbase set ---------------------------------------------------------------------------------------------------"""
def color_rknn_classify(colordatafile,ordr_centroidlst, lettermsk, shapemsk):

  shadesofcolordata = json.load(open(colordatafile))
  centroid_value = np.copy(ordr_centroidlst)
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

  letter_predict = colorneigh.predict([centroid_value[0]]) 
  shape_predict = colorneigh.predict([centroid_value[1]])   
  
  print('letter:', letter_predict,'shape:', shape_predict)

  return {"letter": letter_predict,"shape":shape_predict}

"""color masking from kmeans------------------------------------------------------------------------------------------------------------"""
def letter_shape_mask(reconfiguredimg, ordered_centroids):
  color_range = 1
  letter_upper_rgb = np.array(ordered_centroids[0][0:3]) #used to be 0
  letter_low_rgb = letter_upper_rgb.copy()
  sp_upper_rgb = np.array(ordered_centroids[1][0:3])  #used to be 2
  sp_low_rgb = sp_upper_rgb.copy()
  bckgrd_upper_rgb = np.array(ordered_centroids[2][0:3])  #used to be 2
  bckgrd_low_rgb = bckgrd_upper_rgb.copy()

  letter_upper_rgb[letter_upper_rgb <255] += color_range
  letter_low_rgb[letter_low_rgb >0] -= color_range
  bckgrd_upper_rgb[bckgrd_upper_rgb <255] += color_range
  bckgrd_low_rgb[bckgrd_low_rgb >0] -= color_range   
  sp_upper_rgb[sp_upper_rgb <255] += color_range
  sp_low_rgb[sp_low_rgb >0] -= color_range    


  lettercolormask = cv2.inRange(reconfiguredimg, letter_low_rgb,letter_upper_rgb)
  letterresult = cv2.bitwise_and(reconfiguredimg,reconfiguredimg,mask=lettercolormask)
  letterresult = cv2.bitwise_not(letterresult)
  letterresult = cv2.cvtColor(letterresult, cv2.COLOR_BGR2GRAY)
  ret,letterfiltered = cv2.threshold(letterresult,254,255,cv2.THRESH_BINARY)

  bckcolormask = cv2.inRange(threemaskimage, bckgrd_low_rgb,bckgrd_upper_rgb)
  bckresult = cv2.bitwise_and(reconfiguredimg,reconfiguredimg,mask=bckcolormask)
  bckresult = cv2.bitwise_not(bckresult)
  bckresult = cv2.cvtColor(bckresult, cv2.COLOR_BGR2GRAY)
  ret1,bckbw = cv2.threshold(bckresult,254,255,cv2.THRESH_BINARY)

  spcolormask = cv2.inRange(threemaskimage, sp_low_rgb,sp_upper_rgb)
  shaperesult = cv2.bitwise_and(reconfiguredimg,reconfiguredimg,mask=spcolormask)
  shaperesult = cv2.bitwise_not(shaperesult)
  shaperesult = cv2.cvtColor(shaperesult, cv2.COLOR_BGR2GRAY)
  ret1,spbw = cv2.threshold(shaperesult,254,255,cv2.THRESH_BINARY)

  return letterfiltered, spbw, bckbw

if __name__ == "__main__":
    
    """to test accuracy, it reads in a test json file that lists photo and correct values, then reports number correct
    color takes in image from shape, preprocess the image, feed it into kmeans, kmeans results are fed into knn
    the results of kmeans will be an ordered list of array with R,G,B,radius and the results are matched with
    least distance being letter and second distance is shape"""
    score = 0
    datatestreport = {}
    test = 0
    sharpen = False

    for index in range(1, 91):

        image = cv2.imread(str(index)+'-k50.jpg')
        img = image
        h, w = image.shape[0], image.shape[1]

        if sharpen is True:
            kernel = np.array([[0, -1, 0],
                    [-1, 5.7,-1],
                    [0, -1, 0]])
            sharp_image = cv2.filter2D(src=img, ddepth=-1, kernel=kernel)
   #         cv2.imwrite(str(index)+'Sharpened5-3.jpg', sharp_image)

        rgbimage = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

        threemaskimage, centroid = color_quantization(rgbimage,3, index)








'''     mask1, mask2, mask3 = letter_shape_mask(threemaskimage,centroid) 
        cv2.imwrite(str(index)+"mask1.jpg",mask1)
        cv2.imwrite(str(index)+"mask2.jpg",mask2)
        cv2.imwrite(str(index)+"mask3.jpg", mask3)
        '''
        

        #next is classification
"""     classifiedcolor = color_rknn_classify('colordataset.json',centroid,lettermask,shapemask)

        #classifiedcolor is a dictionary with key 'letter' and 'shape'
        if (rightletter == classifiedcolor["letter"] and rightshape == classifiedcolor["shape"]):
          score += 1
          datatestreport[test].append(score)
          print('correct')
        else:
          datatestreport[test].append([classifiedcolor['letter'].tolist(),classifiedcolor['shape'].tolist()])
        test += 1
        """
        
        
        
