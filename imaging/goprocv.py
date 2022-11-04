import cv2
import time

VID_CAP_PORT = 1
SLEEP_TIME = 2

def saveImages():
    cam = cv2.VideoCapture(VID_CAP_PORT)
    cv2.namedWindow("gopro stream")

    img_counter = 0

    while True:
        ret, frame = cam.read()
        if not ret:
            print("failed to grab frame")
            break
        cv2.imshow("gopro stream", frame)

        k = cv2.waitKey(1)
        if k%256 == 27:
            # ESC pressed
            print("Escape hit, closing...")
            break
        
        time.sleep(SLEEP_TIME)  
        img_name = "opencv_frame_{}.png".format(img_counter)
        cv2.imwrite(img_name, frame)
        print("{} written!".format(img_name))
        img_counter += 1

    cam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    saveImages()