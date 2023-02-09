from types import BuiltinFunctionType
from bs4 import BeautifulSoup
import requests
import re
import colorsys
from pprint import pprint
import json
import numpy as np
import random
from colorir import sRGB

html_string= "https://www.colorsexplained.com/shades-of-"
html_string2= "-color-names/"
html2_hue2 = "-color-names-html-hex-rgb-codes/"
html2_hue = "https://www.color-meanings.com/shades-of-"

colorlist = ["red","orange","yellow","green","blue","purple"]
huecolor = ["white","gray","black","brown"]


dic_lowerbounds = {"red": [np.array([0, 100, 100],np.uint8), np.array([159, 100, 100],np.uint8)], "orange": np.array([10, 100, 150],np.uint8), \
                   "yellow": np.array([20, 100, 100],np.uint8), "green": np.array([36, 100, 100],np.uint8), "blue": np.array([80, 100, 100],np.uint8),\
                   "purple": np.array([132, 100, 100],np.uint8), "gray": np.array([0, 0, 50],np.uint8), \
                   "black": np.array([0, 0, 0],np.uint8), "white": np.array([0, 0, 231],np.uint8), "brown":np.array([10,115,51],np.uint8)
                   } 

dic_upperbounds = {"red": [np.array([9, 255, 255],np.uint8), np.array([179, 255, 255],np.uint8)], "orange": np.array([15, 255, 255],np.uint8), \
                   "yellow": np.array([35, 255, 255],np.uint8), "green": np.array([70, 255, 255],np.uint8), "blue": np.array([128, 255, 255],np.uint8),\
                   "purple": np.array([150, 255, 255],np.uint8), "gray": np.array([180, 18, 120],np.uint8), \
                   "black": np.array([179, 255, 40],np.uint8), "white": np.array([180, 18, 255],np.uint8), "brown":np.array([20,255,127])}

#print(dic_lowerbounds)

twolist = [colorlist,huecolor]

hsv_colorrange = {}
mydata = {}
headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; WOW64; rv:50.0) Gecko/20100101 Firefox/50.0'}

for colorslist in twolist:
    for color in colorslist:
        mydata[color]= []

        if len(colorslist) == 4:
            if color == "brown":
                link = "https://louisem.com/421101/brown-hex-codes"
            else:
                link = html2_hue + color + html2_hue2
        else:
            link = html_string + color + html_string2

        page = requests.get(link, headers=headers)
        html = BeautifulSoup(page.content, 'html.parser')

        html = BeautifulSoup(page.content, 'html.parser')

        
        
        if (color != "brown"): #used to be brown

            for para in html.findAll("p",{"class":"has-white-color has-text-color has-background","class":"has-background"}):
                    stringsss = para.get_text()
                    stringlist = []
                    stringlist = re.split(r'Hex |RGB |CMYK',stringsss)
                    stringlist = [i for i in stringlist if i]

              #      tempdic["Hex"] = stringlist[1]
                    stringlist = stringlist[2].split(',')
                    rgbdataval = [int(stringlist[0].replace(" ","")),int(stringlist[1].replace(" ","")),int(stringlist[2].replace(" ",""))]

                    mydata[color].append(rgbdataval)
        else:
            print('nrowm')
            for para in html.findAll("tr"):
                    stringsss = para.get_text()
                    if (stringsss.find('#')!= -1):

                        stringlist = []
                        stringlist = re.split(r"#|\(",stringsss)
 
               #         tempdic["Hex"] = "#" + stringlist[1]
                        stringlist[2] = stringlist[2].replace(')',"")
  
                        stringlist = stringlist[2].split(',')

                        rgbdataval = [int(stringlist[0].replace(" ","")),  int(stringlist[1].replace(" ","")),  int(stringlist[2].replace(" ","")) ] 
                        
                        mydata[color].append(rgbdataval)

#adding more colors from hsv ranges set delete this section if you only want the webscrape data
#---------------------------------------------------------------------------------------------------------------
        print(color)
        for datagen in range(0,100):
            if color == "red":
                for sett in range(0,2):
                    print(sett)
                    H = random.randint(dic_lowerbounds[color][sett][0],dic_upperbounds[color][sett][0])/179 
                    S = random.randint(dic_lowerbounds[color][sett][1],dic_upperbounds[color][sett][1])/255 
                    V = random.randint(dic_lowerbounds[color][sett][2],dic_upperbounds[color][sett][2])/255 
                    print(color, H,S,V)
                    r,g,b = colorsys.hsv_to_rgb(H,S,V)
                    hsv2rgbvalue = [round(r*255),round(g*255),round(b*255)]
                    mydata[color].append(hsv2rgbvalue)
                    
            else:     
                    H = random.randint(dic_lowerbounds[color][0],dic_upperbounds[color][0])/179 
                    S = random.randint(dic_lowerbounds[color][1],dic_upperbounds[color][1])/255 
                    V = random.randint(dic_lowerbounds[color][2],dic_upperbounds[color][2])/255 
                    print(color, H,S,V)
                    r,g,b = colorsys.hsv_to_rgb(H,S,V)
                    hsv2rgbvalue = [round(r*255),round(g*255),round(b*255)]
                    mydata[color].append(hsv2rgbvalue)
#-------------------------------------------------------------------------------------------------------------


json_object = json.dumps(mydata, indent=4)
with open("colordataset.json", "w") as outfile:
    outfile.write(json_object)





   


#print(nextsegment)
#segmentseg = next(segment)
#print(segmentseg)





#nextsegment = nextsegment.find_all("has-white-color has-text-color has-background")

#nextsegment = nextsegment.find('div')


#for article in nextsegment:
 #   title = article.select('.has-white-color has-text-color has-background')
  #  print(title)

#print(next2segment)

#select_part = segment.select('div.entry-content mvt-content')
#print(select_part)
#print(html.body.div.div.main)
      #.article.div.p.text)
#article = html.select('p.has-white-color has-text-color has-background')

#for article in articles:
# title = article.select('.has-white-color has-text-color has-background)


#/html/body
#/div[2]/div/main/article/div/p[9]