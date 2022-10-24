import generateData
import os
import math
class DataSet():
    def __init__(self, size):
        self.createFilesAndDirectories()
        self.isDistributed = False
        self.size = size
        self.labels = "ABCDEFGHIJKLMNOPQRSTUVWXYZ123456789"
        self.numLabels = 36
        self.distributions = dict()
        self.remaining = 1
        self.generater = generateData.DataGenerator(128)


    def createFilesAndDirectories(self):
        '''Creates file hierarchy needed to run'''
        try:
            os.mkdir('.\dataset')
        except Exception: # directories already exist
            pass
        try:
            os.mkdir('.\dataset\data')
        except Exception: # directories already exist
            pass
        with open(".\dataset\labels.txt", "w") as f:
            f.write("file, label\n")


    def setRemainingDistributions(self):
        '''Sets the remaining distributions given that the customs are filled already'''
        photoDistribution = self.remaining / (self.numLabels - len(self.distributions)) 
        for label in self.labels:
            if(label not in self.distributions):
                self.distributions[label] = photoDistribution
                self.remaining -= photoDistribution
        self.isDistributed = True
    

    def setDistribution(self, key, dist):
        '''Sets a distribution for the given key'''
        if(key in self.distributions):
            self.remaining += self.distributions[key]
        if(0 < dist <= self.remaining):
            self.distributions[key] = dist
            self.remaining -= dist
        else:
            raise Exception("Not enough space for the new distribution: ",self.remaining, dist)


    def generate(self):
        '''Generates dataset and puts it into ./dataset'''
        if not self.isDistributed:
            self.setRemainingDistributions()
        with open("myfonts.txt", "r") as file:
            fonts : list = file.readlines()
        numPhotos = math.ceil(self.size /len(fonts))
        for x in range(len(fonts)):
            self.generater.generate_letters(fonts[x], self.distributions, numPhotos, path =".\dataset\data", labelPath=".\dataset\labels.txt", startNum = (x * numPhotos))
        

def main():
    d = DataSet(2000) # Change the number here to change the size of the dataset
    
    custom_distributions = {}
    #add (key, 0 <= dist <= 1) to the custom_distributions
    #TODO Have them read from a file
    for key, dist in custom_distributions:
         d.setDistribution(key, dist)

    d.setRemainingDistributions()
    d.generate()


    

if __name__ == "__main__":
    main()