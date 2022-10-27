import pygame
import os
import DataGenerator


def gen_fonts():
    try:
        with open("fonts.txt" ,"w") as file:
            for x in pygame.font.get_fonts():
                file.write(x)
                file.write("\n")
    except Exception:
        pass


def main():
    ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ123456789"
    
    try:
        os.mkdir("./fonts") # Make directory if it doesn't exist
    except:
        pass
    
    gen_fonts()

    gen = DataGenerator.DataGenerator()
    with open("fonts.txt", 'r') as f:
        fonts = f.readlines()
    for font in fonts:
        font = font.strip()
        try:
            os.mkdir(f"./fonts/{font}") # Make directories for each font
        except:
            pass
        gen.generate_font_letters(font)
        

    write = open("myfonts.txt", "w")
    with open("fonts.txt", "r") as file:
        f = file.readlines()

    pygame.init()
    resolution = 1048
    screen = pygame.display.set_mode([resolution,resolution])
    
    for fontName in f: # Loop thru and add sample photos for each font for each letter
        fontName = fontName.strip()

        running = True
        while running:
            for x in range(7):
                for y in range(5):
                    imp = pygame.image.load(f".\\fonts\\{fontName}\{ALPHABET[x * 5 + y]}.jpg")
                    screen.blit(imp, (x*128,y*128))
                    running2 = True
            pygame.display.flip()
            for i in pygame.event.get():
                if i.type == pygame.QUIT:
                    pygame.quit()
                    write.close()
                    quit()
                if i.type == pygame.KEYDOWN:
                    if i.key == pygame.K_y:
                        print("Y Pressed")
                        write.write(fontName + "\n")
                        running = False
                    if i.key == pygame.K_n:
                        print("N pressed")
                        running = False
    pygame.quit()


if __name__ == "__main__":
    main()