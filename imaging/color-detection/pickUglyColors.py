import pygame
import os
import ColorGenerator
import json
import random
import time

def get_color_db():
    try:
        with open("uglycolors.json" ,"r") as file:
            color_db = json.load(file)
            return color_db
    except Exception as e:
        print(e)


def main():
    
    # try:
    #     os.mkdir("./fonts") # Make directory if it doesn't exist
    # except:
    #     pass
    
    colors = get_color_db()



    gen = ColorGenerator.ColorGenerator(colors)

    # with open("fonts.txt", 'r') as f:
    #     fonts = f.readlines()
    # for font in fonts:
    #     font = font.strip()
    #     try:
    #         os.mkdir(f"./fonts/{font}") # Make directories for each font
    #     except:
    #         pass
    #     gen.generate_font_letters(font)
        

    # write = open("myfonts.txt", "w")
    # with open("fonts.txt", "r") as file:
    #     f = file.readlines()

    # pygame.init()
    # resolution = 1048
    # screen = pygame.display.set_mode([resolution,resolution])
    
    # for fontName in f: # Loop thru and add sample photos for each font for each letter
    #     fontName = fontName.strip()
    generate = 0 #generate = 1 gens new data, generate == 0 prints color folders
    if(generate == 1):
        COLORS = ["white", "brown", "black", "gray"]
        running = True
        while running:
            newColor = gen.generate_new_color("brown", 0)

            pygame.event.clear()
            i = pygame.event.wait()
            if i.type == pygame.QUIT:
                pygame.quit()
                with open("uglycolors.json", "w") as file:
                    json.dump(colors, file, indent=1)
                quit()
            if i.type == pygame.KEYDOWN:
                if i.key == pygame.K_q:
                    pygame.quit()
                    with open("uglycolors.json", "w") as file:
                        json.dump(colors, file, indent=2)
                    quit()
                elif i.key == pygame.K_w:
                    print("adding to white")
                    colors["white"].append(newColor)
                elif i.key == pygame.K_b:
                    print("adding to brown")
                    colors["brown"].append(newColor)
                elif i.key == pygame.K_g:
                    print("adding to gray")
                    colors["gray"].append(newColor)
                elif i.key == pygame.K_k:
                    print("adding to black")
                    colors["black"].append(newColor)

                elif i.key == pygame.K_f:
                    print("adding to nothing")
                    # colors["green"].append(newColor)
            time.sleep(0.25)
    else:
        gen.gen_color_folders()
    pygame.quit()


if __name__ == "__main__":
    main()