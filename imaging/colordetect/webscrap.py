from types import BuiltinFunctionType
from bs4 import BeautifulSoup
import requests
import re
import colorsys
from pprint import pprint
import json
import numpy as np

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

mydata = {}
headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; WOW64; rv:50.0) Gecko/20100101 Firefox/50.0'}

for colorslist in twolist:
    for color in colorslist:
        if len(colorslist) == 4:
          #  if color == "brown":
          #      link = "https://louisem.com/421101/brown-hex-codes"
          #  else:
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
        if (color != "yes"): #used to be brown
            for para in html.findAll("p",{"class":"has-white-color has-text-color has-background","class":"has-background"}):
                    stringsss = para.get_text()
                    tempdic = {}
                    list = []
                    list = re.split(r'Hex |RGB |CMYK',stringsss)
                    list = [i for i in list if i]
                   # print(list)
                    tempdic["Hex"] = list[1]
                    list = list[2].split(',')
                #    print(list)
                    tempdic['R'] = int(list[0].replace(" ",""))
                    tempdic['G'] = int(list[1].replace(" ",""))
                    tempdic['B'] = int(list[2].replace(" ",""))
             #filter colors based on hue...
                    H,S,V = colorsys.rgb_to_hsv(tempdic['R']/255,tempdic['G']/255,tempdic['B']/255)
                #    print(H)


 #                   if (color not in  huecolor):
 #                       if ((H*179)>dic_upperbounds[color][0] or (179*H)<dic_lowerbounds[color][0]):  
                  #          or (S*255)>dic_upperbounds[color][1] or (255*S)<dic_lowerbounds[color][1] 
                  #          or (V*255)>dic_upperbounds[color][2] or (255*V)<dic_lowerbounds[color][2]):
 #                           continue
                        #    print(H*180)
                    #     print("filtered")
#                        if (color == "red"):
#                            if ( (179*H)<dic_lowerbounds["secondred"][0]):
                 #               or (S*255)>dic_upperbounds["secondred"][1] or (255*S)<dic_lowerbounds["secondred"][1] 
                 #               or (V*255)>dic_upperbounds["secondred"][2] or (255*V)<dic_lowerbounds["secondred"][2]):
    #                            print("didn't pass")
#                                continue
#                            else:
#                                print('high red value')
                  #      else:        
                  #          continue
                    
                    tempdic['H'],  tempdic['S'],  tempdic['V'] = round(H*179),  round(S*255),  round(V*255)    
                    mydata[color].append(tempdic)

        else:
            for para in html.findAll("tr"):
                    stringsss = para.get_text()
                    print(stringsss)
                    if (stringsss.find('#')!= -1):
                        tempdic = {}
                        list = []
                        list = re.split(r"#|\(",stringsss)
                  #      print(list)
                        tempdic["Hex"] = "#" + list[1]
                        list[2] = list[2].replace(')',"")
                  #      print(list[2])
                        list = list[2].split(',')
                        tempdic['R'] = int(list[0].replace(" ",""))
                        tempdic['G'] = int(list[1].replace(" ",""))
                        tempdic['B'] = int(list[2].replace(" ",""))
                        tempdic['H'],  tempdic['S'],  tempdic['V'] = colorsys.rgb_to_hsv( tempdic['R']/255,  tempdic['G']/255,  tempdic['B']/255)
                        tempdic['H'],  tempdic['S'],  tempdic['V'] = round(tempdic['H']*179),  round(tempdic['S']*255),  round(tempdic['V']*255)            
                        mydata[color].append(tempdic)

     

for each in mydata:
    print(each)
    print(len(mydata[each]))


json_object = json.dumps(mydata, indent=4)
with open("sample.json", "w") as outfile:
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