import pygame
import os
ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890"

# Create file directories 
for x in ALPHABET:
    try:
        os.mkdir(f'./data/{x}')
    except:
        pass
pygame.init()
clock = pygame.time.Clock()
resolution = 128


def main():
    # Create file directories 
    for x in ALPHABET:
        try:
            os.mkdir(f'./data/{x}')
        except:
            pass
    # choose = int(input("Please enter num font: "))
    choose = 2
    with open("fonts.txt", "r") as file:
        f = file.readlines()
    font = f[choose].strip()
    print(f"you have chosen: {font}")
    generate_data(font)


def generate_data(font, changeDeg = 15, path=""):
    pygame.init()
    resolution = 128
    changeDeg = 15
    currentDeg = 0
    numPhotos = 360 // changeDeg
    font = pygame.font.SysFont(font, resolution)
    screen = pygame.display.set_mode([resolution,resolution])
    
    for x in ALPHABET: 
        text = x
        for number in range(numPhotos):

            screen.fill((0,0,0)) # Fill screen with black
            letter = font.render(text, True, (255,255,255), (0,0,0))
            letter = pygame.transform.rotate(letter, currentDeg)
            currentDeg += changeDeg
            textRect = letter.get_rect()
            textRect.center = (resolution // 2, resolution // 2)
            screen.blit(letter, textRect)
            pygame.display.flip()
            pygame.image.save(screen, f"{path}/data/{text}/{number}.jpg")
    pygame.quit()
