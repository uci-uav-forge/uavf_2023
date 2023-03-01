from threading import Thread
import time
import sys
def imagingPipeline():
    global img, running
    print("Starting imaging pipeline")
    while running:
        if img != None:
            print("Imaging Pipeline: Got a new image")
            time.sleep(2)
            print("Imaging Pipeline: finished processing image")
            img = None
    print("Imaging Pipeline: Received shutdown")
    sys.exit(0)
def navPipeline():
    global running, img
    print("Starting nav pipeline")
    while running:
        time.sleep(3)
        print("Nav Pipeline: Took a new photo with camera")
        img = "hello"
    print("Nav Pipeline: Received shutdown")

img = None
running = True
imaging = Thread(target=imagingPipeline)
nav = Thread(target =navPipeline)
try:
    imaging.start() 
    nav.start()
    while(True):
        pass 
except (KeyboardInterrupt, SystemExit):
    running = False
    print("Main: Interrupt received. Closing threads")
    imaging.join()
    nav.join()
    print("Main: All threads terminated")
    sys.exit(0)