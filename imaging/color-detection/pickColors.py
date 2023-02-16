import pygame
import os
import ColorGenerator
import json
import random
import time

def get_color_db():
    try:
        with open("colors.json" ,"r") as file:
            color_db = json.load(file)
            return color_db
    except Exception as e:
        print(e)


def main():
    colors = get_color_db()
    gen = ColorGenerator.ColorGenerator(colors)
    generate = 1 #generate = 1 gens new data, generate == 0 prints color folders
    if(generate == 0):
        COLORS = ["red", "orange", "yellow", "green", "blue", "purple"] #"white", "brown", "black", "gray"]
        running = True
        while running:
            newColor = gen.generate_new_color(COLORS[random.randint(0,len(COLORS) - 1)], 0)

            pygame.event.clear()
            i = pygame.event.wait()
            if i.type == pygame.QUIT:
                pygame.quit()
                with open("colors.json", "w") as file:
                    json.dump(colors, file, indent=1)
                quit()
            if i.type == pygame.KEYDOWN:
                if i.key == pygame.K_q:
                    pygame.quit()
                    with open("colors.json", "w") as file:
                        json.dump(colors, file, indent=2)
                    quit()
                elif i.key == pygame.K_r:
                    print("adding to red")
                    colors["red"].append(newColor)
                elif i.key == pygame.K_o:
                    print("adding to orange")
                    colors["orange"].append(newColor)
                elif i.key == pygame.K_y:
                    print("adding to yellow")
                    colors["yellow"].append(newColor)
                elif i.key == pygame.K_g:
                    print("adding to green")
                    colors["green"].append(newColor)
                elif i.key == pygame.K_b:
                    print("adding to blue")
                    colors["blue"].append(newColor)
                elif i.key == pygame.K_p:
                    print("adding to purple")
                    colors["purple"].append(newColor)
                elif i.key == pygame.K_p:
                    print("adding to purple")
                    colors["purple"].append(newColor)
                elif i.key == pygame.K_f:
                    print("adding to nothing")
                    # colors["green"].append(newColor)
            time.sleep(0.25)
    else:
        gen.gen_color_folders()
    pygame.quit()


if __name__ == "__main__":
    main()