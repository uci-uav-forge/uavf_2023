from fieldcapturer import *
import time
import argparse

ap = argparse.ArgumentParser()
ap.add_argument('interval',type=int)
ar = ap.parse_args()


fc = FieldCapturer()

while True:
    img = fc.capture()
    time.sleep(ar.interval)

