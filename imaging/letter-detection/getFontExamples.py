import generateData
import os
with open("fonts.txt", 'r') as f:
    fonts = f.readlines()
try:
    os.mkdir("./fonts")
except:
    pass
gen = generateData.DataGenerator()
for font in fonts:
    font = font.strip()
    try:
        os.mkdir(f"./fonts/{font}")
    except:
        pass
    gen.generate_font_letters(font)
    