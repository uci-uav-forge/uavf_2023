from types import BuiltinFunctionType
from bs4 import BeautifulSoup
import requests
import re
import colorsys
from pprint import pprint
import json
import numpy as np
from colorir import sRGB

html_string= "https://www.colorsexplained.com/shades-of-"
html_string2= "-color-names/"
html2_hue2 = "-color-names-html-hex-rgb-codes/"
html2_hue = "https://www.color-meanings.com/shades-of-"

colorlist = ["red","orange","yellow","green","blue","purple"]
huecolor = ["white","gray","black","brown"]

dic_lowerbounds = {"red": np.array([0, 70, 100],np.uint8),"secondred": np.array([159, 70, 100],np.uint8), "orange": np.array([11, 70, 150],np.uint8), \
                   "yellow": np.array([24, 70, 100],np.uint8), "green": np.array([36, 70, 100],np.uint8), "blue": np.array([80, 70, 100],np.uint8),\
                   "purple": np.array([132, 70, 100],np.uint8), "gray": np.array([0, 0, 50],np.uint8), \
                   "black": np.array([0, 0, 0],np.uint8), "white": np.array([0, 0, 231],np.uint8),
                   } 

dic_upperbounds = {"red": np.array([9, 255, 255],np.uint8),"secondred": np.array([180, 255, 255],np.uint8), "orange": np.array([20, 255, 255],np.uint8), \
                   "yellow": np.array([35, 255, 255],np.uint8), "green": np.array([70, 255, 255],np.uint8), "blue": np.array([128, 255, 255],np.uint8),\
                   "purple": np.array([150, 255, 255],np.uint8), "gray": np.array([180, 18, 150],np.uint8), \
                   "black": np.array([180, 255, 40],np.uint8), "white": np.array([180, 18, 255],np.uint8)}

#print(dic_lowerbounds)

twolist = [colorlist,huecolor]

hsv_colorrange = {}
mydata = {}
headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; WOW64; rv:50.0) Gecko/20100101 Firefox/50.0'}

for colorslist in twolist:
    for color in colorslist:
        if len(colorslist) == 4:
            if color == "brown":
                link = "https://louisem.com/421101/brown-hex-codes"
            else:
                link = html2_hue + color + html2_hue2
        else:
            link = html_string + color + html_string2

    #    print(link)
        page = requests.get(link, headers=headers)
#soup = BeautifulSoup(html_page.content, 'html.parser')
      #  print(page.text)
#warning = soup.find('div',class = "wrapper wrapper-content")



        html = BeautifulSoup(page.content, 'html.parser')

     #   print(html)
#segment = html.body

#nextsegment = segment.select('div.content > div > main > article > div')
#mydata = nextsegment[0].split("<p>")
#mydata = nextsegment[0].split('<p>')

      #  print(colorslist)
        #print('why u change')
     #   print(len(colorslist))

        html = BeautifulSoup(page.content, 'html.parser')
        mydata[color] = []
        upper=[0,0,0]
        lower=[255,255,255]
        if (color != "brown"): #used to be brown
            for para in html.findAll("p",{"class":"has-white-color has-text-color has-background","class":"has-background"}):
                    stringsss = para.get_text()
                    if color == 'blue':
                        print(stringsss)
                 #   mydata[color].append([stringsss])
                    tempdic = {}
                    list = []
                    list = re.split(r'Hex |RGB |CMYK',stringsss)
                    list = [i for i in list if i]
                   # print(list)
                    tempdic["Hex"] = list[1]
                    list = list[2].split(',')
                #    print(list)
      #              tempdic['R'] = int(list[0].replace(" ",""))
      #              tempdic['G'] = int(list[1].replace(" ",""))
      #              tempdic['B'] = int(list[2].replace(" ",""))
                    tempdic['RGB'] = [int(list[0].replace(" ","")),int(list[1].replace(" ","")),int(list[2].replace(" ",""))]

             #filter colors based on hue...
                    H,S,V = colorsys.rgb_to_hsv(int(list[0].replace(" ",""))/255,int(list[1].replace(" ",""))/255,int(list[2].replace(" ",""))/255)
                    h,s,v = sRGB(tempdic['RGB'][0], tempdic['RGB'][1], tempdic['RGB'][2]).hsv(round_to=1)
                #    tempdic['HSV'] = [round(H*179),  round(S*255),  round(V*255) ]  
                    tempdic['HSV'] = [h,  s,  v ]
                    tempdic['raw'] = list
                    if color == 'blue':
                        print(h,s,v)
                        print(tempdic) 
                    mydata[color].append(tempdic)

        else:
            print('nrowm')
            for para in html.findAll("tr"):
                    stringsss = para.get_text()
               #     mydata[color].append([stringsss])
                    if (stringsss.find('#')!= -1):
                        tempdic = {}
                        list = []
                        list = re.split(r"#|\(",stringsss)
                  #      print(list)
                        tempdic["Hex"] = "#" + list[1]
                        list[2] = list[2].replace(')',"")
                  #      print(list[2])
                        list = list[2].split(',')

                        tempdic['RGB'] = [int(list[0].replace(" ","")),  int(list[1].replace(" ","")),  int(list[2].replace(" ","")) ] 
                        H,S,V = colorsys.rgb_to_hsv( tempdic['RGB'][0]/255,  tempdic['RGB'][1]/255,  tempdic['RGB'][2]/255)
                        
 
                        tempdic['HSV'] = [round(H*179),  round(S*255),  round(V*255) ]   
                        tempdic['raw'] = list
                #        print(tempdic)          
                        mydata[color].append(tempdic)

        #calculating the range

colorrange = {}     

for each in mydata:
    temph = []
    temps = []
    tempv = []
    print(each)
    for eachcolorvalue in mydata[each]:
    #    print(eachcolorvalue)
        temph.append(eachcolorvalue['HSV'][0])
        temps.append(eachcolorvalue['HSV'][1])
        tempv.append(eachcolorvalue['HSV'][2])
        temph = sorted(temph)
        temps = sorted(temps)
        tempv = sorted(tempv)
     #   print(temph)
  #  print('why', temph)
    colorrange[each] = [[temph[0],temps[0],tempv[0]],[temph[-1],temps[-1],tempv[-1]]]

   # print(each)
   # print(len(mydata[each]))

print(colorrange)

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