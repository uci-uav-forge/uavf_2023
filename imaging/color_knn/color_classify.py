import os
import json

from sklearn.neighbors import RadiusNeighborsClassifier

COLOR_KNN_PATH = os.path.dirname(os.path.realpath(__file__))

class ColorClassifier:
    def __init__(self, kradius = 100):
        shadesofcolordata = json.load(open(f"{COLOR_KNN_PATH}/traincolors.json"))

        xdata = []
        ylabel = []

        for colorlabel in shadesofcolordata:
            for shadevalue in shadesofcolordata[colorlabel]:
                xdata.append(shadevalue) 
                ylabel.append(colorlabel)

        self.classifier = RadiusNeighborsClassifier(radius = kradius, weights = "distance",outlier_label = "gray")
        self.classifier.fit(xdata, ylabel)
    def predict(self, rgb_val, bgr=False):
        return self.classifier.predict([rgb_val if not bgr else list(reversed(rgb_val))])[0]
    

if __name__ == "__main__":
    test_val = (255,0,0)
    classifier = ColorClassifier()
    print(classifier.predict(test_val))