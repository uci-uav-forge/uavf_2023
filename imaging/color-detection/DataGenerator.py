import pygame
import os
import math

ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ123456789"
import random
REDS = [(230,20,20), (190, 20, 20), (193,20,55), (240,60,60)]
ORANGES = [(240, 160, 20), (200, 100, 20), (230, 120,20), (200,90,20)]
BLUES = [(20,20,200), (30,30,120), (20,20,230), (40,40,200)]
GREENS = [(30, 200, 30), (20,180,20), (40, 230, 120) ,(20,150,20)]
fields = ["./field1.png","./field2.png","./field3.png"]
colors = [REDS, ORANGES, BLUES, GREENS]
class DataGenerator():
    
    
    def __init__(self, resolution=128, path=""):
        try:
            os.mkdir(f'.{path}')
        except Exception:
            pass
        try:
            os.mkdir(f'.{path}/dataset')
        except Exception as e:
            print(e)
        try:
            os.mkdir(f'.{path}/dataset/data')
        except Exception:
            pass
        self.resolution = 128
        self.path = path
        pygame.init()
        self.screen = pygame.display.set_mode([self.resolution,resolution])
    def rand1(self):
        return random.randint(-self.resolution//16, self.resolution//4)
    
    def rand2(self):
        return random.randint(-self.resolution//16, self.resolution//8)
    
    def generate_letters(self, numImages,):
        num = 0
        
        with open(f".{self.path}/dataset/labels.txt", "a") as file:
            currentDeg = 0
            screen = pygame.display.set_mode([self.resolution,self.resolution])
            for index, x in enumerate(ALPHABET): 
                numPhotos = math.ceil(numImages)
                currentDeg = 0
                for _ in range(numPhotos):
                    text = ALPHABET[random.randint(0,34)]
                    currentDeg += random.randint(-50, 65)
                    screen.fill((0,0,0)) # Fill screen with black

                    imp = pygame.image.load(fields[random.randint(0,2)]).convert()
                    imp = pygame.transform.rotate(imp, random.randint(0,4) * 90)
                    screen.blit(imp, (random.randint(self.resolution-250,0), random.randint(self.resolution-250,0)))
                    color = [*colors[random.randint(0,3)][random.randint(0,3)]]
                    color = [color[0] + random.randint(-10,10),color[1] + random.randint(-10,10),color[2] + random.randint(-10,10)]
                    shape = _ % 3
                    match shape:
                        case 0:
                            shape = pygame.Rect(10,30,self.resolution//3 + self.rand1(), self.resolution//3 + self.rand1())
                            shape.center = (self.resolution // 2 + self.rand2(), self.resolution // 2 + self.rand2())
                            pygame.draw.rect(screen, color, shape)
                        case 1:
                            shape = pygame.Rect(10,30,self.resolution//2 + self.rand1(), self.resolution//2 + self.rand1())
                            shape.center = (self.resolution // 2 + self.rand2(), self.resolution // 2 + self.rand2())
                            pygame.draw.ellipse(screen, color, shape)
                            
                        case 2:
                            pygame.draw.polygon(screen, color, [(self.resolution*0.25 + self.rand2(),self.resolution* 0.25 + self.rand2()), (self.resolution*0.75+self.rand2(),self.resolution* 0.25+self.rand2()),(self.resolution*0.75+ self.rand2(),self.resolution*0.75 + self.rand2()),(self.resolution*0.25+self.rand2(),self.resolution*0.75+self.rand2())])
                    font = pygame.font.SysFont("arial", self.resolution//3)
                    letter = font.render(text, True, (255,255,255))
                    letter = pygame.transform.rotate(letter, currentDeg)
                    textRect = letter.get_rect()
                    if(type(shape) is not int):
                        textRect.center = shape.center
                    else:
                        textRect.center =  (self.resolution // 2, self.resolution // 2)
                    screen.blit(letter,textRect)
                    pygame.display.flip()
                    pygame.image.save(screen, f".{self.path}/dataset/data/{num}.jpg")
                    file.write(f"./data/{num}.jpg" + ", " + str(index) + "\n")
                    num += 1
        return num


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
