import pygame

ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ123456789"
pygame.init()
clock = pygame.time.Clock()
resolution = 1048
screen = pygame.display.set_mode([resolution,resolution])

running = True
write = open("myfonts.txt", "w")
with open("fonts.txt", "r") as file:
    f = file.readlines()
for fontName in f:
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