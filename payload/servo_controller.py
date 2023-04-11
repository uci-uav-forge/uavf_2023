from adafruit_servokit import ServoKit

# max range of servo in degrees
servoMax = 26

#We're using a 16 channel servo controller
kit = ServoKit(channels=16)

#Initializes the servo controller
def setupController():
    for i in range(16):
        kit.servo[i].actuation_range = servoMax
        kit.servo[0].set_pulse_width_range(1000, 2000)

#Open specified servo
def openServo(servoIndex : int):
    kit.servo[servoIndex].angle = servoMax

#Close specified servo
def closeServo(servoIndex : int):
    kit.servo[servoIndex].angle = 0

#Open all servos
def openAllServos():
    for i in range(5):
        kit.servo[i].angle = servoMax

#Close all servos
def closeAllServos():
    for i in range(5):
        kit.servo[i].angle = 0

class ServoController:
    def __init__(self):
        setupController()
    def openServo(self, servoIndex : int):
        openServo(servoIndex)
    def closeServo(self, servoIndex : int):
        closeServo(servoIndex)
    def openAllServos(self):
        openAllServos()
    def closeAllServos(self):
        closeAllServos()

if __name__=="__main__":
    servoController = ServoController()
    servoController.openAllServos()
    input("Press enter to close all servos")
    servoController.closeAllServos()
    do_open_servos = input("Do you want to open all servos again? (y/n) ")
    if do_open_servos == "y":
        servoController.openAllServos()
    print("Servo test done")
    