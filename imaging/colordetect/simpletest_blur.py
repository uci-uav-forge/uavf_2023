from types import BuiltinFunctionType
#from bs4 import BeautifulSoup
#import requests
#import re
import colorsys
from pprint import pprint
import json
import numpy as np
import cv2
import shutil
import colorsys


#"next": np.array([159, 50, 70],np.uint8),

# tuple written as b,g,r

dic_colors = {"red": (0,0,255), "orange": (26,118, 245), \
                   "yellow": (0,255,255), "green": (0, 255, 0), "blue": (255,0, 0),\
                   "purple": (255,8,135), "brown": (38,58,79), "gray": (110,110, 110), \
                   "black": (0, 0, 0), "white": (255, 255, 255),
                   } 

colors = ["red", "orange", "yellow", "green", "blue", "purple", "gray", "brown", "black", "white"]
hsv_dic_color = {}
for eachcolor in dic_colors:
    h,s,v = colorsys.rgb_to_hsv(dic_colors[eachcolor][2]/255,dic_colors[eachcolor][1]/255,dic_colors[eachcolor][0]/255)
   # print(h,s,v)
    hsv_dic_color[eachcolor] = [round(h*179),round(s*255),round(v*255)]


print(hsv_dic_color)

mydata = {}

testsetname = '\circletestset1\\'
setname = 'Ccircletestset1'
newfolder = '\\blurtests'


imagebasepath = r'C:\Users\kirva\Desktop\forge\sample.jpg'
savefilepath = 'C:\\Users\\kirva\\Desktop\\forge\\winter2023'
baseimage = cv2.imread(imagebasepath)
row,col,channel = baseimage.shape

center_coordinates = (col//2, row//2)
ksize = (20,20)
thickness = [-1, 15]
radius = [row//2, row//4]
total = 1
for x in range(len(colors)):
    if x == len(colors)-1:
        break
    else:
        for y in range(x, len(colors)):
            if y != x:
                tempdic = {}
                tempdic2 = {}
                color1 = dic_colors[colors[y]] 
                color2 = dic_colors[colors[x]]
    #            img = cv2.rectangle(image, (20,20), (row-20,col-20), color1, thickness[0])
    #            img = cv2.rectangle(img, (50,50), (row-50,col-50), color2, thickness[1])

    # we are making circles for shape and letter
                img = cv2.circle(baseimage, center_coordinates, radius[0], color1, thickness[0])
                img = cv2.circle(img, center_coordinates, radius[1], color2, thickness[1])
                img = cv2.blur(img,ksize)
                name = 'circleblursample-' + str(total) + colors[y]+ colors[x] + '.jpg'
                
                cv2.imwrite(name, img)
   #             shutil.move(r'C:\Users\kirva\Desktop\forge\winter2023\testgeneratecode\\' +name,r'C:\Users\kirva\Desktop\forge\winter2023\nonblurtest\circletestset1\\'+ name)
                tempdic['shape'] = colors[y]
                tempdic['letter'] = colors[x]
      #          print(name,tempdic)
     #           mydata[savefilepath+ newfolder + testsetname+ name ] = tempdic
                mydata[ name ] = tempdic
                total +=1
 #               img = cv2.rectangle(image, (20,20), (row-20,col-20), color2, thickness[0])
  #              img = cv2.rectangle(img, (50,50), (row-50,col-50), color1, thickness[1])
                img = cv2.circle(baseimage, center_coordinates, radius[0], color2, thickness[0])
                img = cv2.circle(img, center_coordinates, radius[1], color1, thickness[1])
                img = cv2.blur(img,ksize)
                name2 = 'circleblursample-' + str(total) + colors[x]+ colors[y] + '.jpg'
                cv2.imwrite(name2, img)
         #       shutil.move(savefilepath + r'\testgeneratecode\\' +name2,savefilepath + newfolder+ testsetname + name2)
                tempdic2['shape'] = colors[x]
                tempdic2['letter'] = colors[y]
   #             print(name2,tempdic2)
   #             mydata[savefilepath+ newfolder + testsetname+ name2 ] = tempdic2
                mydata[ name2 ] = tempdic2
    #            print(mydata)
                total += 1
              #  break
    #break
                



json_object = json.dumps(mydata,indent=4)
with open("circleblurtestset1.json", "w") as outfile:
    outfile.write(json_object)
#shutil.move(savefilepath + r'\testgeneratecode\circletestset1.json',savefilepath + newfolder+ testsetname + 'circletestset1.json')
