import urllib
from io import StringIO, BytesIO
import binascii
from math import log, exp, tan, atan, pi, ceil
from PIL import Image
import urllib.request

EARTH_RADIUS = 6378137
EQUATOR_CIRCUMFERENCE = 2 * pi * EARTH_RADIUS
INITIAL_RESOLUTION = EQUATOR_CIRCUMFERENCE / 256.0
ORIGIN_SHIFT = EQUATOR_CIRCUMFERENCE / 2.0

def latlontopixels(lat, lon, zoom):
    mx = (lon * ORIGIN_SHIFT) / 180.0
    my = log(tan((90 + lat) * pi/360.0))/(pi/180.0)
    my = (my * ORIGIN_SHIFT) /180.0
    res = INITIAL_RESOLUTION / (2**zoom)
    px = (mx + ORIGIN_SHIFT) / res
    py = (my + ORIGIN_SHIFT) / res
    return px, py

def pixelstolatlon(px, py, zoom):
    res = INITIAL_RESOLUTION / (2**zoom)
    mx = px * res - ORIGIN_SHIFT
    my = py * res - ORIGIN_SHIFT
    lat = (my / ORIGIN_SHIFT) * 180.0
    lat = 180 / pi * (2*atan(exp(lat*pi/180.0)) - pi/2.0)
    lon = (mx / ORIGIN_SHIFT) * 180.0
    return lat, lon

############################################

upperleft = '38.3163534, -76.551114'

#
#

zoom = 22   # be careful not to get too many images!

apiKey = "api key"

############################################

ullat, ullon = map(float, upperleft.split(','))

# Set some important parameters
scale = 1
maxsize = 640

# convert all these coordinates to pixels
ulx, uly = latlontopixels(ullat, ullon, zoom)

# calculate total pixel dimensions of final image
dx, dy = 5312, 2988

print(dx,dy,maxsize)

# calculate rows and columns
cols, rows = int(ceil(dx/maxsize)), int(ceil(dy/maxsize))



# calculate pixel dimensions of each small image
bottom = 120
largura = int(ceil(dx/cols))
altura = int(ceil(dy/rows))
alturaplus = altura + bottom


final = Image.new("RGB", (int(dx), int(dy)))
for x in range(cols):
    for y in range(rows):
        dxn = largura * (0.5 + x)
        dyn = altura * (0.5 + y)
        latn, lonn = pixelstolatlon(ulx + dxn, uly - dyn - bottom/2, zoom)
        position = ','.join((str(latn), str(lonn)))
        print(x, y, position)
        urlparams = urllib.parse.urlencode({'center': position,
                                      'zoom': str(zoom),
                                      'size': '%dx%d' % (largura, alturaplus),
                                      'maptype': 'satellite',
                                      'sensor': 'false',
                                      'scale': scale,
                                      'key':apiKey})
        url = 'https://maps.google.com/maps/api/staticmap?' + urlparams
        f=urllib.request.urlopen(url)
        im=Image.open(BytesIO(f.read()))
        final.paste(im, (int(x*largura), int(y*altura)))
final.show()
final.save('fieldgrab.png')

##TODO: Grab area of field that is appropriate to area GoPro captures
# Capture height info??
# Integrate with shape recognition datagen.
