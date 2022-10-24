import pygame
import os
import math
ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ123456789"

# # Create file directories 
# for x in ALPHABET:
#     try:
#         os.mkdir(f'./data/{x}')
#     except:
#         pass
# pygame.init()
# clock = pygame.time.Clock()
# resolution = 128


# def main():
#     # Create file directories 
#     for x in ALPHABET:
#         try:
#             os.mkdir(f'./data/{x}')
#         except:
#             pass
#     # choose = int(input("Please enter num font: "))
#     choose = 2
#     with open("fonts.txt", "r") as file:
#         f = file.readlines()
#     font = f[choose].strip()
#     print(f"you have chosen: {font}")
#     generate_letters(font)

class DataGenerator():
    
    def __init__(self, resolution=128):
        self.resolution = 128
        pygame.init()
        self.screen = pygame.display.set_mode([self.resolution,resolution])


    def generate_font_letters(self, fontName):
        '''Generates letters for the given font'''
        font = pygame.font.SysFont(fontName, self.resolution)
        for x in ALPHABET:
            self.screen.fill((0,0,0)) # Fill screen with black
            letter = font.render(x, True, (255,255,255), (0,0,0))
            textRect = letter.get_rect()
            textRect.center = (self.resolution // 2, self.resolution // 2)
            self.screen.blit(letter, textRect)
            pygame.display.flip()
            pygame.image.save(self.screen, f".\\fonts\\{fontName}\\{x}.jpg")
    

    def generate_letters(self, font, distribution, numImages, path="", labelPath=".\dataset\labels.txt", startNum = 0):
        with open(labelPath, "a") as file:
            currentDeg = 0
            font = pygame.font.SysFont(font, self.resolution)
            screen = pygame.display.set_mode([self.resolution,self.resolution])
            
            for index, x in enumerate(ALPHABET): 
                text = x
                numPhotos = math.ceil(distribution[text] * numImages)
                for _ in range(numPhotos):
                    
                    changeDeg = 360 / numPhotos
                    screen.fill((0,0,0)) # Fill screen with black
                    letter = font.render(text, True, (255,255,255), (0,0,0))
                    letter = pygame.transform.rotate(letter, currentDeg)
                    currentDeg += changeDeg
                    textRect = letter.get_rect()
                    textRect.center = (self.resolution // 2, self.resolution // 2)
                    screen.blit(letter, textRect)
                    pygame.display.flip()
                    pygame.image.save(screen, f"{path}\{startNum}.jpg")
                    file.write(f"{startNum}.jpg" + ", " + str(index) + "\n")
                    startNum += 1


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
