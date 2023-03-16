import pygame
import os
import math
ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ123456789"
import random
class DataGenerator():
    
    
    def __init__(self, resolution=32, path=""):
        self.resolution = 128
        self.path = path
        pygame.init()
        self.screen = pygame.display.set_mode([self.resolution,resolution])


    def generate_font_letters(self, fontName):
        '''Generates letters for the given font'''
        font = pygame.font.SysFont(fontName, self.resolution//4)
        for x in ALPHABET:
            self.screen.fill((0,0,0)) # Fill screen with black
            letter = font.render(x, True, (255,255,255), (0,0,0))
            textRect = letter.get_rect()
            textRect.center = (self.resolution // 2, self.resolution // 2)
            self.screen.blit(letter, textRect)
            pygame.display.flip()
            pygame.image.save(self.screen, f"./fonts/{fontName}/{x}.jpg")
    

    def generate_letters(self, font, distribution, numImages, startNum = 0):
        with open(f".{self.path}/dataset/labels.txt", "a") as file:
            currentDeg = 0
            font = pygame.font.SysFont(font, self.resolution)
            screen = pygame.display.set_mode([self.resolution,self.resolution])
            
            for index, x in enumerate(ALPHABET): 
                text = x
                numPhotos = math.ceil(distribution[text] * numImages)
                currentDeg = 0
                for _ in range(numPhotos):
                    currentDeg += random.randint(-50, 65)
                    screen.fill((0,0,0)) # Fill screen with black
                    letter = font.render(text, True, (255,255,255), (0,0,0))
                    letter = pygame.transform.rotate(letter, currentDeg)
                    textRect = letter.get_rect()
                    textRect.center = (self.resolution // 2, self.resolution // 2)
                    screen.blit(letter, textRect)
                    pygame.display.flip()
                    pygame.image.save(screen, f".{self.path}/dataset/data/{startNum}.jpg")
                    file.write(f"data/{startNum}.jpg" + ", " + str(index) + "\n")
                    startNum += 1
        return startNum


    # def generate_data(self, font, numImages = 15, path=""):
    #     '''Generates data for the given font and number of images'''
    #     pygame.init()
    #     changeDeg = 360 // numImages
    #     currentDeg = 0
    #     numPhotos = 360 // changeDeg
    #     font = pygame.font.SysFont(font, self.resolution)
    #     screen = pygame.display.set_mode([self.resolution,self.resolution])
        
    #     for x in ALPHABET: 
    #         text = x
    #         for number in range(numPhotos):

    #             screen.fill((0,0,0)) # Fill screen with black
    #             letter = font.render(text, True, (255,255,255), (0,0,0))
    #             letter = pygame.transform.rotate(letter, currentDeg)
    #             currentDeg += changeDeg
    #             textRect = letter.get_rect()
    #             textRect.center = (self.resolution // 2, self.resolution // 2)
    #             screen.blit(letter, textRect)
    #             pygame.display.flip()
    #             pygame.image.save(screen, f"{path}/data/{text}/{number}.jpg")
    #     pygame.quit()
