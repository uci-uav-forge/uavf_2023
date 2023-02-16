import pygame
import os
import math
ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ123456789"
import random
class ColorGenerator():
    
    
    def __init__(self, color_db, resolution=128, path=""):
        self.resolution = 128
        self.color_db = color_db
        self.path = path
        pygame.init()
        self.screen = pygame.display.set_mode([self.resolution,resolution])


    def generate_new_color(self, color, isRandom=0):
        if(not isRandom):
            print(f"generating from {color}")
            base_color = self.color_db[color][random.randint(0,len(self.color_db[color])-1)] # take a random known color
            new_color = [base_color[0] + random.randint(-15,15), base_color[1]+ random.randint(-15,15), base_color[2] + random.randint(-15,15)]
            for index in range(3):
                while(new_color[index] < 0 or new_color[index] > 255):
                    new_color[index] = base_color[index] + random.randint(-15,15)
        else:
            new_color = [random.randint(0,255), random.randint(0,255), random.randint(0,255)]
        self.screen.fill(new_color) # Fill screen with color
        pygame.display.flip()
        return new_color

    def gen_color_folders(self):
        try:
            os.mkdir(f'./data/')
        except Exception as e:
            pass
        for color in self.color_db.keys():
            try:
                os.mkdir(f'./data/{color}')
            except Exception as e:
                print(e)
            for c in self.color_db[color]:
                print(color, c)
                self.screen.fill(c)
                pygame.display.flip()
                pygame.image.save(self.screen, f"./data/{color}/{c[0]},{c[1]},{c[2]}.jpg")


    

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
