import pygame
import os
ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890"

def gen_fonts():
    try:
        with open("fonts.txt" ,"w") as file:
            for x in pygame.font.get_fonts():
                file.write(x)
                file.write("\n")
    except Exception:
        pass

def main():
    gen_fonts()
    pygame.init()
    clock = pygame.time.Clock()
    resolution = 128
    # choose = int(input("Please enter num font: "))

    with open("fonts.txt", "r") as file:
        f = file.readlines()

    screen = pygame.display.set_mode([resolution,resolution])
    for fontName in f:
        fontName = fontName.strip()
        try:
            os.mkdir(f'./data/fonts/{fontName}')
        except:
            pass
        font = pygame.font.SysFont(fontName, resolution)
        for x in ALPHABET: 
            text = x
            screen.fill((0,0,0)) # Fill screen with black
            letter = font.render(text, True, (255,255,255), (0,0,0))
            textRect = letter.get_rect()
            textRect.center = (resolution // 2, resolution // 2)
            screen.blit(letter, textRect)
            pygame.display.flip()
            pygame.image.save(screen, f"./data/fonts/{fontName}/{x}.jpg")
pygame.quit()


if __name__ == "__main__":
    main()