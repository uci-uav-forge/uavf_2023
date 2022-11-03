from fieldcapturer import *
import time

interval = 2

fc = FieldCapturer()

while True:
    img = fc.capture()
    time.sleep(2)

